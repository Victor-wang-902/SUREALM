import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention, BertLayer, BertEncoder, BertModel

import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import (
    logging
)
from Embedding import ContextEmbed
import warnings
logger = logging.get_logger(__name__)


class RetrievalSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_keys: Optional[torch.Tensor] = None,
        hidden_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_keys: Optional[torch.FloatTensor] = None,
        encoder_hidden_values: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        #print("states", hidden_states.shape)
        #print("keys", hidden_keys.shape)
        #print("value", hidden_values.shape)
        #print("atnmsk", attention_mask.shape)
        mixed_query_layer = self.query(hidden_states)
        '''print("hidden_states", hidden_states)
        print("hidden_keys", hidden_keys)
        print("hidden_values", hidden_values)
        print("attention_mask", attention_mask)
        print("encoder_hidden_states", encoder_hidden_states)
        print("encoder_hidden_keys", encoder_hidden_keys)
        print("encoder_hidden_values", encoder_hidden_values)
        print("encoder_attention_mask", encoder_attention_mask)
        '''
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.

        # extension for cross-attention-like retrieval attention 
        is_cross_attention = encoder_hidden_states is not None or encoder_hidden_keys is not None
        is_retrieval = encoder_hidden_keys is not None or hidden_keys is not None
        #print(is_cross_attention, is_retrieval)
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            # retrieval cross attention
            if is_retrieval:
                key_layer = self.transpose_for_scores(self.key(encoder_hidden_keys))
                value_layer = self.transpose_for_scores(self.value(encoder_hidden_values))
            # normal decoder cross attention
            else:
                key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
                value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            # retrieval self attention
            if is_retrieval:
                key_layer = self.transpose_for_scores(self.key(hidden_keys))
                value_layer = self.transpose_for_scores(self.value(hidden_values))
            # normal self attention
            else:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            # retrieval self attention
            if is_retrieval:
                key_layer = self.transpose_for_scores(self.key(hidden_keys))
                value_layer = self.transpose_for_scores(self.value(hidden_values))
            # normal self attention
            else:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)
        #print("q", query_layer.shape, "k", key_layer.shape)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        #print("score", attention_scores.shape)
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        #print("mask", attention_mask.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #if is_cross_attention:
        #    print("atnscore shape", attention_scores.shape)
        if attention_mask is not None:
            #print(attention_mask)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        #print(attention_probs.shape)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class RetrievalAttention(BertAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.self = RetrievalSelfAttention(config, position_embedding_type=position_embedding_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_keys: Optional[torch.Tensor] = None,
        hidden_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_keys: Optional[torch.FloatTensor] = None,
        encoder_hidden_values: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            hidden_keys,
            hidden_values,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_hidden_keys,
            encoder_hidden_values,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class RetrievalBertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        # self attention layer
        self.attention = RetrievalAttention(config)

        # cross attention layer
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RetrievalAttention(config, position_embedding_type="absolute")

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_keys: Optional[torch.Tensor] = None,
        hidden_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_keys: Optional[torch.FloatTensor] = None,
        encoder_hidden_values: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        # check retrieval inputs are valid
        assert (hidden_keys is None and hidden_values is None) or (hidden_keys is not None and hidden_values is not None)
        assert hidden_keys is None or encoder_hidden_keys is None

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        # self retireval attention pass
        self_attention_outputs = self.attention(
            hidden_states,
            hidden_keys,
            hidden_values,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and (encoder_hidden_states is not None or encoder_hidden_keys is not None):
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            '''print("layer arguments")
            print("atno", attention_output)
            print("atnm", attention_mask)
            print("head_mask", head_mask)
            print("ehs", encoder_hidden_states)
            print("ehk", encoder_hidden_keys)
            print("ehv", encoder_hidden_values)
            print("eatnmsk", encoder_attention_mask)
            print("eatnmsk shape", encoder_attention_mask.shape)
            print("end of leyer")'''
            # cross retrieval attention pass, here decoder key and value can be hard coded None 
            cross_attention_outputs = self.crossattention(
                attention_output,
                None,
                None,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_hidden_keys,
                encoder_hidden_values,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    
class RetrievalDecoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([RetrievalBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_keys: Optional[torch.Tensor] = None,
        hidden_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_keys: Optional[torch.FloatTensor] = None,
        encoder_hidden_values: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    hidden_keys,
                    hidden_values,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_hidden_keys,
                    encoder_hidden_values,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    hidden_keys,
                    hidden_values,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_hidden_keys,
                    encoder_hidden_values,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class RetrievalModel(BertModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True, pre_path=None, post_path=None):
        super().__init__(config, add_pooling_layer)
        if not hasattr(self.config, "add_cross_attention"):
            self.config.add_cross_attention = False
        # add pre computed context embeddings for keys and values
        if not hasattr(self.config, "is_retrieval"):
            self.config.is_retrieval = False
        if pre_path is None: 
            if hasattr(self.config, "pre_path"):
                pre_path = self.config.pre_path
        if post_path is None:
            if hasattr(self.config, "post_path"):
                post_path = self.config.post_path
        if self.config.is_retrieval and (pre_path is None or ctx_value_path is None):
            warnings.warn("Retrieval mode is activated but not all embedding layers are loaded. Either pass external embeddings or define embedding layers.")
        if hasattr(self.config, "ctx_vocab_size"):
            self.init_embeddings(self.config.ctx_vocab_size, self.config.hidden_size)
        self.load_embeddings(pre_path, post_path)
        self.encoder = RetrievalDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()
    
    # method to add retrieval embeddings and set is_retrieval to True post initialization
    def is_retrieval(self, key=None, value=None):
        self.config.is_retrieval = True
        self.load_embeddings(key, value)
        if (not hasattr(self, "ctx_key_embeddings") or not hasattr(self, "ctx_value_embeddings")) and (key is None or value is None):
            warnings.warn("Retrieval mode is activated but not both key embedding layers are initialized. Either pass external embeddings or redefine embedding layers.") 

    def disable_retrieval(self):
        self.config.is_retrieval = False

    def init_embeddings(self, size, dim):
        self.ctx_key_embeddings = ContextEmbed(size=size,dim=dim)
        self.ctx_value_embeddings = ContextEmbed(size=size,dim=dim)

    # method to add retrieval embeddings but not activate is_retrieval
    def load_embeddings(self, key=None, value=None):
        if key is not None:
            self.ctx_key_embeddings = ContextEmbed(key)
            self.config.ctx_vocab_size = self.ctx_key_embeddings.embed.num_embeddings
            assert self.config.hidden_size == self.ctx_key_embeddings.embed.embedding_dim

        if value is not None:
            self.ctx_value_embeddings = ContextEmbed(value)    
            self.config.ctx_vocab_size = self.ctx_value_embeddings.embed.num_embeddings
            assert self.config.hidden_size == self.ctx_value_embeddings.embed.embedding_dim
        if hasattr(self, "ctx_key_embeddings") and hasattr(self, "ctx_value_embeddings"):
            assert self.ctx_key_embeddings.embed.num_embeddings == self.ctx_value_embeddings.embed.num_embeddings

    #method to initialize cross attention weights with self attention within the same layer
    def init_cross_attention_weights(self):
        assert self.config.add_cross_attention == True
        for lay in self.encoder.layer:
            with torch.no_grad():
                lay.crossattention.self.query.weight.copy_(lay.attention.self.query.weight)
                lay.crossattention.self.query.bias.copy_(lay.attention.self.query.bias)

                lay.crossattention.self.key.weight.copy_(lay.attention.self.key.weight)
                lay.crossattention.self.key.bias.copy_(lay.attention.self.key.bias)

                lay.crossattention.self.value.weight.copy_(lay.attention.self.value.weight)
                lay.crossattention.self.value.bias.copy_(lay.attention.self.value.bias)

                lay.crossattention.output.dense.weight.copy_(lay.attention.output.dense.weight)
                lay.crossattention.output.dense.bias.copy_(lay.attention.output.dense.bias)

                lay.crossattention.output.LayerNorm.weight.copy_(lay.attention.output.LayerNorm.weight)
                lay.crossattention.output.LayerNorm.bias.copy_(lay.attention.output.LayerNorm.bias)

    def clear_embeddings(self):
        key = self.ctx_key_embeddings
        value = self.ctx_value_embeddings
        self.ctx_key_embeddings = None
        self.ctx_value_embeddings = None
        return key, value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_keys: Optional[torch.Tensor] = None,
        encoder_hidden_values: Optional[torch.Tensor] = None,
        encoder_indices: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """

        # check if input is valid. here encoder_hidden_keys, encoder_hidden_values, encoder_indices serves as decoder key and value or encoder key and value depending on whether is_cross_attention is True
        if self.config.is_retrieval:
            if encoder_attention_mask is not None:
                assert encoder_attention_mask.dim() == 3
            elif attention_mask is not None:
                assert  attention_mask.dim() == 3
            else:
                raise ValueError("retrieval mask must be explicit")
        else:
            assert encoder_hidden_keys is None and encoder_hidden_values is None and encoder_indices is None
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # retrieval self attention requires explicit attention map
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        
        
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_retrieval:
            if encoder_attention_mask is not None:
                #print("before eatnmsk", encoder_attention_mask)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
                #print("after eatnmsk", encoder_extended_attention_mask)
            else:
                encoder_extended_attention_mask = None
        else:
            if self.config.is_decoder and encoder_hidden_states is not None:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = None


        # if cross retrieval attention, then encoder_attention_mask needs to be explicit and thus can be used directly.
        #if self.config.add_cross_attention and self.config.is_retrieval:
        #    encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]


        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        # pass the right arguments to support retrieval
        if self.config.is_retrieval:
            if encoder_indices is not None:
                ctx_key_embedding_output = self.ctx_key_embeddings(encoder_indices)
                ctx_value_embedding_output = self.ctx_value_embeddings(encoder_indices)
            elif encoder_hidden_keys is not None:
                ctx_key_embedding_output = encoder_hidden_keys
                ctx_value_embedding_output = encoder_hidden_values
            if self.config.add_cross_attention:
                hidden_keys = None
                hidden_values = None
                if self.config.concat_self:
                    encoder_hidden_keys = torch.concat((embedding_output, ctx_key_embedding_output), dim=1)
                    encoder_hidden_values = torch.concat((embedding_output, ctx_value_embedding_output), dim=1)
                else:
                    encoder_hidden_keys = ctx_key_embedding_output
                    encoder_hidden_values = ctx_value_embedding_output

            else:
                hidden_keys = torch.concat((embedding_output, ctx_key_embedding_output), dim=1)
                hidden_values = torch.concat((embedding_output, ctx_value_embedding_output), dim=1)
                encoder_hidden_keys = None
                encoder_hidden_values = None
        else:
            hidden_keys = None
            hidden_values = None
            encoder_hidden_keys = None
            encoder_hidden_values = None
        '''
        print("model arguments")
        print("embo", embedding_output)
        print("hk", hidden_keys)
        print("hv", hidden_values)
        print("atnmsk", extended_attention_mask)
        print("enhs", encoder_hidden_states)
        print("enhk", encoder_hidden_keys)
        print("enhv", encoder_hidden_values)
        print("enatnmsk", encoder_extended_attention_mask)
        print("end of head arguments")'''
        encoder_outputs = self.encoder(
            embedding_output,
            hidden_keys=hidden_keys,
            hidden_values=hidden_values,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_keys=encoder_hidden_keys,
            encoder_hidden_values=encoder_hidden_values,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )





'''class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, ):
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.txt_emb = 
    
    def forward(self, tgt, ctx, tgt_ctx_mask=None, tgt_ctx_key_padding_mask=None):
        output = tgt
        context = ctx
        for mod in self.layers:
            output = mod(output, ctx, tgt_ctx_mask=tgt_ctx_mask, tgt_ctx_key_padding_mask=tgt_ctx_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        return output
'''

        