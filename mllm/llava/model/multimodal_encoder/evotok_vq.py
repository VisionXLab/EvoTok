from torch import nn
import torch
import torch.nn.functional as F
import math
from functools import partial
import torch.distributed as distributed
from einops import rearrange
from transformers.modeling_utils import get_parameter_dtype
import sys
from pathlib import Path
cur_file_path = Path(__file__).resolve()
sys.path.append(str(cur_file_path.parent.parent.parent.parent.parent))
from evotok.tokenizer.vq_model import VQ_models
import types

def no_op_train(self, mode=True):
    pass

def freeze_module_eval(module, except_modules=None, prefix=""):
    if except_modules is None:
        except_modules = []

    module_name = prefix.rstrip(".")
    if module_name not in except_modules:
        module.eval()
        module.training = False
        for p in module.parameters():
            p.requires_grad = False
        module.train = lambda mode=True: module

        for name, child in module.named_children():
            child_prefix = f"{module_name}.{name}" if module_name else name
            freeze_module_eval(child, except_modules, prefix=child_prefix)


def evotok_model(
    model_name, codebook_size, teacher,
    codebook_embed_dim=32,
    vqgan_depth=4,
    vqkd_depth=16,
    pretrain_path=None,
):
    vq_model = VQ_models[model_name](
        codebook_size=codebook_size,
        codebook_embed_dim=codebook_embed_dim,
        vqgan_depth=vqgan_depth,
        vqkd_depth=vqkd_depth,
        teacher=teacher
    )
    vq_model.train = types.MethodType(no_op_train, vq_model)
    freeze_module_eval(vq_model, except_modules=[])

    if pretrain_path is not None:
        print("evotok load from:", pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        if "ema" in checkpoint:  # ema
            model_weight = checkpoint["ema"]
        elif "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            raise Exception("please check model weight")
        missing, unexpected = vq_model.load_state_dict(model_weight, strict=False)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

        print("evotok model load success!!")
        vq_model.eval()
        vq_model.training = False
        vq_model.quantize.training = False
        vq_model.quantize.codebooks.training = False

    return vq_model