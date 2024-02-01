#python inference.py --model_path v10/35 --knowledge_path fast_processed_data/dstc_2gram_top3 --interactive --max_length 20
#python inference.py --model_path v10/35 --knowledge_path fast_processed_data/dstc_2gram_top3 --prompt_file dummy_inference_text.txt --max_length 20
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from models_gpt2 import RetrievalGPT2Model

class RetrievalGenerationModelGPT2(GPT2LMHeadModel):
    def __init__(self, config, args=None):
        super().__init__(config)
        if not args.not_retrieval:
            self.transformer = RetrievalGPT2Model(config)
        self.tokenizer = args.tokenizer
        self.is_retrieval = not args.not_retrieval
        self.post_init()

    def post_post_init(self, args=None, freeze_backbone=None, max_length=None, max_context_length_per_k=None): # if the model is initialized from huggingface pretrained model without cross attention, call this function after from_pretrained
        if args is not None:
            freeze_backbone = args.freeze_backbone
            max_length = args.max_length
            max_context_length_per_k = args.max_context_length_per_k

        if freeze_backbone is not None and freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False
        if max_length is not None:
            self.config.max_length = max_length
        if max_context_length_per_k is not None:
            self.config.max_context_length_per_k = max_context_length_per_k

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_keys: Optional[torch.Tensor] = None,
        encoder_hidden_values: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.is_retrieval:
            transformer_outputs = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_keys=encoder_hidden_keys,
                encoder_hidden_values=encoder_hidden_values,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            transformer_outputs = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is None:
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            num_tokens = torch.sum(torch.ne(labels, -100)).item()
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        to_return = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        to_return["num_tokens"] = torch.tensor(num_tokens, dtype=torch.int32).to(self.device)
        return to_return
    
    def save_pretrained(self, output_path, save_head=False, save_emb=False):
        self.tokenizer.save_pretrained(output_path)
        
        if save_head:
            
            super().save_pretrained(output_path)
                
        else:
            
            self.transformer.save_pretrained(output_path)
                