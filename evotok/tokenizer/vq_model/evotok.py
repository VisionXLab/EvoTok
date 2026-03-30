import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)
import math
import random
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from dataclasses import dataclass, field
from timm.models.layers import trunc_normal_
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from transformers.modeling_utils import get_parameter_device, get_parameter_dtype
from ..quantizer import SharedResidualQuantizer
from .pixel_model import Decoder
from .vqkd_model import VisionTransformer, ScalingLayerForSigLip




def copy_new_embedding(old_embedding, requires_grad=True):
    new_embedding = nn.Embedding(old_embedding.weight.size(0), old_embedding.weight.size(1))
    new_embedding.weight = nn.Parameter(old_embedding.weight.clone())
    new_embedding.weight.requires_grad = requires_grad
    return new_embedding

def drop_scale(original_scales, num_to_drop=1):
    """
    Randomly remove scales from scale list.
    
    Args:
        original_scales: list of scales
        num_to_drop: Number of scales to randomly remove (default 1)
        
    Returns:
        New scale list
    """
    if num_to_drop >= len(original_scales) - 1:
        raise ValueError("Cannot drop that many items")
    
    drop_candidates = list(range(1, len(original_scales)))
    indices_to_drop = set(random.sample(drop_candidates, num_to_drop))
    return [item for i, item in enumerate(original_scales) if i not in indices_to_drop]


@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 32
    codebook_show_usage: bool = True
    restart_unused_codes: bool = True
    vqgan_depth: int = 4
    vqkd_depth: int = 16
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0
    teacher: str = None # option: ["siglip_256", "siglip_384"]
    infer_interpolate: bool = False
    vq_warmup: int = 0

def get_model_default_params():
    return dict(img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dim=1152, depth=12, num_heads=12,  
                mlp_ratio=4., qkv_bias=True,  qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                norm_layer='LayerNorm', init_values=0., use_abs_pos_emb=True, use_rel_pos_bias=False,
                use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001)


class EvoTok(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        # vqgan decoder
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

        self.teacher = config.teacher

        encoder_config, decoder_config = get_model_default_params(), get_model_default_params()
        if self.teacher == 'siglip_384':
            img_size = 27 * 2**4 # 384
            self.decoder_out_dim = 1152
        elif self.teacher == 'siglip_256':
            encoder_config["embed_dim"] = decoder_config["embed_dim"] = 1024
            img_size = 256
            self.decoder_out_dim = 1024
        else:
            raise NotImplementedError
        
        semantic_code_dim = config.codebook_embed_dim
            
        encoder_config['img_size'] = img_size
        encoder_config['num_classes'] = 0

        # decoder settings
        decoder_config['img_size'] = img_size // decoder_config['patch_size']
        decoder_config['patch_size'] = 1
        decoder_config['in_chans'] = config.codebook_embed_dim
        decoder_config['num_classes'] = 0
        decoder_config['depth'] = 3

        print('Final encoder config', encoder_config)

        if self.teacher == 'siglip_384':
            self.encoder_vqkd = SiglipVisionModel.from_pretrained("google/siglip2-so400m-patch14-384")
            for name, param in self.encoder_vqkd.named_parameters():
                if 'head' in name:
                    param.requires_grad = False
                if 'encoder.layers.26' in name:
                    param.requires_grad = False
                if 'post_layernorm' in name:
                    param.requires_grad = False
        elif self.teacher == 'siglip_256':
            self.encoder_vqkd = SiglipVisionModel.from_pretrained("google/siglip2-large-patch16-256")
            for name, param in self.encoder_vqkd.named_parameters():
                if 'head' in name:
                    param.requires_grad = False
                if 'encoder.layers.23' in name:
                    param.requires_grad = False
                if 'post_layernorm' in name:
                    param.requires_grad = False
        else:
            raise ValueError(f'teacher {self.teacher} not supported')
        
        print('Final decoder config', decoder_config)
        self.decoder_vqkd = VisionTransformer(**decoder_config)

        ### task layer ###
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], semantic_code_dim) # for quantize
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)
        # scaling layers
        if self.teacher in ['siglip_384', 'siglip_256']:
            self.scaling_layer = ScalingLayerForSigLip()
        else:
            raise ValueError(f'teacher {self.teacher} not supported')        

        ### parameters required by llava ###
        self.code_dim = config.codebook_embed_dim
        self.embed_dim = self.decoder_out_dim
        self.n_embed = config.codebook_size
        self.compression = 2**(len(config.encoder_ch_mult) - 1)
        ### parameters required by llava ###
        
        # quantizer
        self.quantize = SharedResidualQuantizer(
            config.codebook_size, self.code_dim, 
            code_depth=(config.vqgan_depth, config.vqkd_depth),
            show_usage=config.codebook_show_usage,
            restart_unused_codes=config.restart_unused_codes,
            vq_warmup=config.vq_warmup
        )

        print(f'Current model is: EvoTok with encoder {self.teacher}, initialization finished.')
    
    def clone_vq_codebook(self, requires_grad):
        cloned_vqkd_embedding = copy_new_embedding(self.quantize.embedding_vqkd, requires_grad)
        cloned_vqgan_embedding = copy_new_embedding(self.quantize.embedding_vqgan, requires_grad)
        return (cloned_vqkd_embedding, cloned_vqgan_embedding)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def device(self):
        return get_parameter_device(self)

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    def encode(self, x):
        if self.teacher == "clipb_224":
            vqkd_feature = self.encoder_vqkd(x, return_patch_tokens=True)
            vqkd_feature = self.encode_task_layer(vqkd_feature.type_as(self.encode_task_layer[-1].weight))
            N = vqkd_feature.shape[1]
            B = vqkd_feature.shape[0]
            h, w = int(math.sqrt(N)), int(math.sqrt(N))
            vqkd_feature = rearrange(vqkd_feature, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        elif self.teacher == 'vitamin_xlarge_256':
            vqkd_x = self.scaling_layer(x)
            # vqkd
            vqkd_feature = self.encoder_vqkd.forward_features(vqkd_x)
            vqkd_feature = self.encode_task_layer(vqkd_feature.type_as(self.encode_task_layer[-1].weight))
            N = vqkd_feature.shape[1]
            h, w = int(math.sqrt(N)), int(math.sqrt(N))
            vqkd_feature = rearrange(vqkd_feature, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        elif self.teacher in ['siglip_384', 'siglip_256']:
            vqkd_x = self.scaling_layer(x)
            vqkd_feature = self.encoder_vqkd(vqkd_x, output_hidden_states=True).hidden_states[-2]
            vqkd_feature = self.encode_task_layer(vqkd_feature.type_as(self.encode_task_layer[-1].weight))
            N = vqkd_feature.shape[1]
            h, w = int(math.sqrt(N)), int(math.sqrt(N))
            vqkd_feature = rearrange(vqkd_feature, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        else:
            raise ValueError(f'teacher {self.teacher} not supported')

        quant, emb_loss, info = self.quantize(vqkd_feature)
        return quant, emb_loss, info

    def decode(self, quant):
        vqgan_quant, vqkd_quant = quant
        vqkd_recon, vqgan_recon = None, None

        if vqkd_quant is not None:
            vqkd_recon = self.decoder_vqkd(vqkd_quant, return_patch_tokens=True)
            vqkd_recon = self.decode_task_layer(vqkd_recon)
        if vqgan_quant is not None:
            vqgan_recon = self.post_quant_conv(vqgan_quant)
            vqgan_recon = self.decoder(vqgan_recon)

            if self.teacher == 'siglip_384':
                vqgan_recon = F.interpolate(vqgan_recon, size=(384, 384), mode='bicubic')

        dec = (vqkd_recon, vqgan_recon)
        return dec

    @torch.no_grad()
    def get_code_emb_with_depth(self, code):
        return self.quantize.embed_code_with_depth(code)


    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################

def EvoTokFunc(**kwargs):
    return EvoTok(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))
