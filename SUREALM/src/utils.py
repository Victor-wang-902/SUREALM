import torch
import torch.nn.functional as F


class AttentionMaskGenerator:  
    '''
    Attention mask generator used in data collator
    '''
    def __init__(self, topk=1, padding="right"):
        self.topk = topk

    def attention_mask(self, inputs):  # Generate retrieval attention mask based on encoded inputs from tokenizer
        seq_len = inputs["input_ids"].shape[1]
        ctx_len = inputs["encoder_indices"].shape[1] if "encoder_indices" in inputs.keys() else inputs["encoder_hidden_keys"].shape[1]
        ngram = inputs["ngram"]
        upper = self.triu_with_stride(seq_len, seq_len)
        lower = self.triu_with_stride(ctx_len, seq_len, ngram, self.topk, False)
        mask = torch.concat((upper, lower), 0)[None,:,:]
        return mask

    def __call__(self, inputs):  # Combine padding mask with attention mask.
        padding_mask = self.pad_mask(inputs).type(torch.int32)
        attn_mask = self.attention_mask(inputs).type(torch.int32).to(padding_mask.device)
        mask_t = padding_mask & attn_mask
        mask = mask_t.transpose(-1,-2)
        return mask

    @staticmethod
    def pad_mask(inputs):  # padding mask for retrieval attention
        def convert_pad_mask(original, text_len):
            return original.unsqueeze(-1).repeat(1, 1, text_len)
        if "encoder_indices" in inputs.keys():
            indices = inputs["encoder_indices"].detach().clone()
            indices[indices != 0] = 1
        elif "encoder_hidden_keys" in inputs.keys() and "encoder_hidden_values" in inputs.keys():
            indices = inputs["encoder_hidden_keys"].detach().clone()
            indices = indices[:,:,0]
            indices[indices != 0] = 1
            indices = indices.type(torch.int32)
        else:
            raise Exception("something bad happened")
        text_len = inputs["attention_mask"].shape[1]
        upper = convert_pad_mask(inputs["attention_mask"], text_len)
        lower = convert_pad_mask(indices, text_len)
        return torch.concat([upper, lower], dim=1)

    @staticmethod
    def triu_with_stride(h, w, ngram=1, topk=1, is_upper=True):  # generate attention mask according to `ngram` and `topk`
        diagonal = 1 if not is_upper else 0
        mask = torch.ones(w,w)
        mask = torch.triu(mask, diagonal=diagonal)
        mask = torch.repeat_interleave(mask, topk,dim=0)
        mask = torch.repeat_interleave(mask, ngram, dim=1)
        mask = mask[:h,:w]
        return mask


def mean_pooling_norm(embeddings, attention_mask):  
    '''
    mean pooling from Huggingface sentence-transformers with normalization
    '''
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    embeddings = torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return F.normalize(embeddings, p=2, dim=1)


def mean_pooling(model_output, attention_mask):  
    '''
    mean pooling without normalization
    '''
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def normalize_np(embs):  
    '''
    normalization
    '''
    return F.normalize(torch.tensor(embs), p=2, dim=1).numpy()


def weighted_pooling(emb, attention_mask, window_size=8, dropoff=0.):  
    '''
    implementation of truncation window and dropoff
    '''
    if emb.dim() == 2:
        emb_exp = emb.unsqueeze(0)
    else:
        emb_exp = emb.clone()
    mask = torch.full(emb_exp.shape, dropoff).to(attention_mask.device)
    mask[:,:window_size,:] = 1.
    mask = torch.cumprod(mask, dim=1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(emb_exp.size()).float()
    mask = mask * input_mask_expanded
    emb_exp = torch.sum(emb_exp * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    return F.normalize(emb_exp, p=2, dim=1)


def encode(inputs, args=None, model=None):  
    '''
    encoding using sentence transformers. Adapted from Huggingface sentence-transformers
    '''
    if args is not None:
        model = args.model
    with torch.no_grad():
        model_output = model(**inputs, return_dict=True)  # Compute token embeddings
    embeddings = mean_pooling(model_output, inputs['attention_mask'])  # Perform pooling
    embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize embeddings
    return embeddings


def encode_no_pool(inputs, args=None, model=None):  
    '''
    encoding using sentence transformers without mean pooling.
    '''
    if args is not None:
        model = args.model
    with torch.no_grad():
        model_output = model(**inputs, return_dict=True)
    token_embeddings = model_output.last_hidden_state
    return token_embeddings
    
