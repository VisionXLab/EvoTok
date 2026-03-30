import warnings
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from einops import rearrange
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN
from llava.mm_utils import tokenize_conversation


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.vocab_size = config.vocab_size
        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        print("Using LLaVAQwen")

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        image_ids=None,
        cfg=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )
        # prepare for generation
        input_ids_copy = input_ids.clone() if input_ids is not None else None
        if image_ids is not None:
            self.model.vision_tower.vision_tower.eval()
            if self.is_image_or_video_start(input_ids[-1, -1]):
                inputs_embeds = self.get_model().embed_tokens(input_ids)
            else:
                image_ids_end = image_ids[-1].clone()
                inputs_embeds = self.model.vision_tower.rqtransformer.embed_with_model_aux(image_ids_end, self.model.vision_tower.vision_tower)
                inputs_embeds = torch.cumsum(inputs_embeds, dim=-2)[:,:,-1,:]
                inputs_embeds = self.get_model().mm_projector(inputs_embeds)
                
            input_ids = None
            assert not self.training, "this code is only for generation"

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if dpo_forward:
            raise NotImplementedError

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        if image_ids is not None:
            assert not self.training, "this code is only for generation"
            self.model.vision_tower.rqtransformer.eval()
            self.model.vision_tower.vision_tower.eval()
            if self.is_image_or_video_start(input_ids_copy[-1, -1]):
                hidden_state =  hidden_states[:, -1, :]
                if len(hidden_state.shape) == 2:
                    hidden_state = hidden_state.unsqueeze(1)
                image_hidden_state, code = self.model.vision_tower.rqtransformer.generate(hidden_state, self.model.vision_tower.vision_tower, cfg)
                image_ids.append(code)
                image_hidden_state = self.get_model().mm_projector(image_hidden_state)
                hidden_states = torch.cat([hidden_states[:, :-1, :], image_hidden_state], dim=1)
            else:
                image_hidden_state, code = self.model.vision_tower.rqtransformer.generate(hidden_states, self.model.vision_tower.vision_tower, cfg)
                image_ids.append(code)
                image_hidden_state = self.get_model().mm_projector(image_hidden_state)
                hidden_states = image_hidden_state

            loss = None
            logits = self.lm_head(hidden_states)
            logits = logits.float()

        image_hidden_states = []
        image_labels = []
        noimage_labels = []

        if labels is not None:
            for i in range(hidden_states.shape[0]):
                label = labels[i]
                hidden_state = hidden_states[i]
                label_zero = label[:, 0].clone()

                if self.config.mm_use_vi_start_end:
                    image_start_index = torch.nonzero(torch.eq(label_zero, self.vocab_size - 4)).squeeze(1)
                    image_end_index = torch.nonzero(torch.eq(label_zero, self.vocab_size - 3)).squeeze(1)
                    video_start_index = torch.nonzero(torch.eq(label_zero, self.vocab_size - 2)).squeeze(1)
                    video_end_index = torch.nonzero(torch.eq(label_zero, self.vocab_size - 1)).squeeze(1)
                    image_start_index = torch.cat([image_start_index, video_start_index])
                    image_end_index = torch.cat([image_end_index, video_end_index])
                else:
                    image_start_index = torch.nonzero(torch.eq(label_zero, self.vocab_size - 2)).squeeze(1)
                    image_end_index = torch.nonzero(torch.eq(label_zero, self.vocab_size - 1)).squeeze(1)

                assert len(image_start_index) == len(image_end_index), f"length of image_start_index is {len(image_start_index)}, length of image_end_index is {len(image_end_index)}"

                if len(image_start_index) > 0:
                    for start_idx, end_idx in zip(image_start_index, image_end_index):
                        image_label = label[start_idx+1:end_idx, :]
                        image_labels.append(image_label)
                        image_hidden_state = hidden_state[start_idx:end_idx-1, :]
                        image_hidden_states.append(image_hidden_state)
                        label_zero[start_idx+1:end_idx] = -100

                noimage_labels.append(label_zero)
            
            # For video
            image_hidden_states_aux = []
            image_labels_aux = []
            image_hidden_states_length = [img.shape[0] for img in image_hidden_states]
            image_hidden_states_length_relative = [img // min(image_hidden_states_length) for img in image_hidden_states_length]
            for l in range(len(image_hidden_states_length_relative)):
                if image_hidden_states_length_relative[l] > 1:
                    image_hidden_states_aux += torch.split(image_hidden_states[l], min(image_hidden_states_length), dim=0)
                    image_labels_aux += torch.split(image_labels[l], min(image_hidden_states_length), dim=0)
                else:
                    image_hidden_states_aux.append(image_hidden_states[l])
                    image_labels_aux.append(image_labels[l])

            if len(image_hidden_states_aux) > 0:
                image_hidden_states = torch.stack(image_hidden_states_aux, 0)
                image_labels = torch.stack(image_labels_aux, 0)

            noimage_labels = torch.stack(noimage_labels, 0)


        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss_fct = CrossEntropyLoss()

        loss = None
        image_loss = None

        if len(image_hidden_states) > 0 and torch.is_tensor(image_hidden_states):
            if hasattr(self.model.vision_tower, "rqtransformer"):
                outs = self.model.vision_tower.rqtransformer(image_hidden_states, image_labels - self.vocab_size, self.model.vision_tower.vision_tower)
            else:
                raise NotImplementedError()
            B, seq_len, D, C = outs.shape
            image_logits = outs.reshape(B*seq_len*D, C).contiguous()
            image_labels = image_labels.reshape(B*seq_len*D).contiguous() - self.config.vocab_size
            image_loss = loss_fct(image_logits, image_labels)

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = noimage_labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if image_loss is not None:
            loss = loss + image_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def is_image_or_video_start(self, idx):
        if self.config.mm_use_vi_start_end:
            return idx == self.vocab_size - 2 or idx == self.vocab_size - 4
        else:
            return idx == self.vocab_size - 2

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)


    @torch.inference_mode()
    def generate_image_content(self, tokenizer, prompt: str, cfg: float = 3.0, generation_nums: int = 1, temperature: float = 0) -> torch.Tensor:
        input_ids_list = []

        conversation = [{"from": "human", "value": prompt}]
        input_ids = tokenize_conversation(conversation, tokenizer, add_generation_prompt=True, image_generation=True).cuda()
        input_ids_list += [input_ids] * generation_nums

        cfg_conversation = [{"from": "human", "value": " "}]
        cfg_input_ids = tokenize_conversation(cfg_conversation, tokenizer, add_generation_prompt=True, image_generation=True).cuda()
        input_ids_list += [cfg_input_ids] * generation_nums

        max_length = max([len(input_ids) for input_ids in input_ids_list])
        input_ids = torch.zeros((len(input_ids_list), max_length), dtype=input_ids_list[0].dtype).cuda()
        attention_mask = torch.zeros((len(input_ids_list), max_length)).bool().cuda()
        for i in range(len(input_ids_list)):
            input_ids[i, -len(input_ids_list[i]):] = input_ids_list[i]
            attention_mask[i, -len(input_ids_list[i]):] = True

        image_ids = []
        outputs = super().generate(inputs=input_ids, attention_mask=attention_mask,
                                  max_new_tokens=self.model.vision_tower.num_patches,
                                  temperature=temperature, do_sample=True if temperature > 0 else False,
                                  use_cache=True, return_dict_in_generate=True, output_hidden_states=True,
                                  image_ids=image_ids, cfg=cfg)
        
        image_ids = torch.cat(image_ids, dim=1)

        image_embeds = self.model.vision_tower.rqtransformer.embed_with_model_aux(image_ids, self.model.vision_tower.vision_tower)
        image_embeds = torch.cumsum(image_embeds, dim=-2)[:,:,-1,:]
        num_patches = int(self.model.vision_tower.num_patches_per_side)
        image_embeds = rearrange(image_embeds, 'b (w h) d -> b d w h', w=num_patches, h=num_patches)
        
        _, response = self.model.vision_tower.vision_tower.decode((image_embeds, None))
        response = response.to(torch.float32).add_(1).mul_(127.5).clamp_(0, 255)

        return response.chunk(2)[0]

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, image_ids=None, cfg=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if image_ids is not None:
            inputs["image_ids"] = image_ids
        if cfg is not None:
            inputs["cfg"] = cfg
        
        return inputs


    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (
                                attention_mask.shape[0],
                                target_shape - attention_mask.shape[1],
                            ),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        if type(images) is list:
            images = torch.cat(images, dim=0)
        elif images.ndim == 5:
            images = images.flatten(0, 1)

        input_image_ids = input_ids[input_ids == IMAGE_TOKEN_INDEX]
        vqgan_features, vqkd_features, tokens = self.encode_images(images, input_image_ids)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids_copy = input_ids.clone()
        input_ids_copy[input_ids_copy == IMAGE_TOKEN_INDEX] = 0
        input_embeds = self.get_model().embed_tokens(input_ids_copy)

        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        input_embeds_1 = [
            cur_input_embeds[cur_attention_mask]
            for cur_input_embeds, cur_attention_mask in zip(input_embeds, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_ids = input_ids[batch_idx]
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # # to handle the case where there are no images in the input
                cur_image_features = vqgan_features[0]
                cur_input_embeds_1 = input_embeds_1[batch_idx]
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx].unsqueeze(1).expand(-1, tokens.shape[-1]))
                continue

            cur_input_embeds = input_embeds_1[batch_idx]
            image_token_indices = (
                [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )
            cur_labels = labels[batch_idx]

            cur_input_ids_noim = []
            cur_labels_noim = []
            cur_input_embeds_no_im = []
            cur_labels_im = [cur_labels[image_token_indices[i]] for i in range(1, len(image_token_indices) - 1)]
            
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_input_embeds_no_im.append(cur_input_embeds[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i].unsqueeze(1).expand(-1, tokens.shape[-1]))
                if i < num_images:
                    cur_tokens = tokens[cur_image_idx]
                    if cur_labels_im[i] != IGNORE_INDEX:
                        # this is for generation task
                        cur_image_features = vqgan_features[cur_image_idx]
                        cur_new_labels.append(cur_tokens)
                    else:
                        # this is for understanding task
                        cur_image_features = vqkd_features[cur_image_idx]
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0], tokens.shape[-1]),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )                      
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            if any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
                warnings.warn("Inputs truncated!")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len, tokens.shape[-1]),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            # new: add image & video tokens
            if model_args.mm_use_vi_start_end:
                num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN], special_tokens=True)
            else:
                num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[:] = embed_tokens_weight[:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def encode_images(self, images, image_ids):
        vqgan_features, vqkd_features, tokens = self.get_model().get_vision_tower()(images, with_codes=True)
        vqgan_features = self.get_model().mm_projector(vqgan_features) if vqgan_features is not None else None
        vqkd_features = self.get_model().mm_projector(vqkd_features) if vqkd_features is not None else None
        tokens = tokens + self.vocab_size   # shift labels

        return vqgan_features, vqkd_features, tokens

AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
