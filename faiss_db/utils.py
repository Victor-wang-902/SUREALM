import numpy as np
import torch
import torch.nn.functional as F

class AttentionMaskGenerator:
    def __init__(self, topk=1, padding="right"):
        
        self.topk = topk

    def attention_mask(self, inputs):
        seq_len = inputs["input_ids"].shape[1]
        ctx_len = inputs["encoder_indices"].shape[1] if "encoder_indices" in inputs.keys() else inputs["encoder_hidden_keys"].shape[1]
        ngram = inputs["ngram"]
        
        upper = self.triu_with_stride(seq_len, seq_len)
        lower = self.triu_with_stride(ctx_len, seq_len, ngram, self.topk, False)
        mask = torch.concat((upper, lower), 0)[None,:,:]
        return mask

    def __call__(self, inputs):
        padding_mask = self.pad_mask(inputs).type(torch.int32)
        #print("padding",padding_mask)
        attn_mask = self.attention_mask(inputs).type(torch.int32).to(padding_mask.device)
        mask_t = padding_mask & attn_mask
        mask = mask_t.transpose(-1,-2)
        return mask


    @staticmethod
    def pad_mask(inputs):
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
    def triu_with_stride(h, w, ngram=1, topk=1, is_upper=True):
        diagonal = 1 if not is_upper else 0
        mask = torch.ones(w,w)
        mask = torch.triu(mask, diagonal=diagonal)
        mask = torch.repeat_interleave(mask, topk,dim=0)
        mask = torch.repeat_interleave(mask, ngram, dim=1)
        mask = mask[:h,:w]
        return mask

class NaiveAttentionMaskGenerator:
    def __init__(self, topk=1, padding="right"):
        
        self.topk = topk

    def attention_mask(self, inputs):
        seq_len = inputs["input_ids"].shape[1]
        ctx_len = inputs["encoder_indices"].shape[1] if "encoder_indices" in inputs.keys() else inputs["encoder_hidden_keys"].shape[1]
        ngram = inputs["ngram"]
        
        lower = self.triu_with_stride(ctx_len, seq_len, ngram, self.topk, False)
        mask = lower[None,:,:]
        return mask

    def __call__(self, inputs):
        padding_mask = self.pad_mask(inputs).type(torch.int32)
        #print("padding",padding_mask)
        attn_mask = self.attention_mask(inputs).type(torch.int32).to(padding_mask.device)
        mask_t = padding_mask & attn_mask
        mask = mask_t.transpose(-1,-2)
        return mask


    @staticmethod
    def pad_mask(inputs):
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
        lower = convert_pad_mask(indices, text_len)
        return lower


        


    @staticmethod
    def triu_with_stride(h, w, ngram=1, topk=1, is_upper=True):
        diagonal = 1 if not is_upper else 0
        mask = torch.ones(w,w)
        mask = torch.triu(mask, diagonal=diagonal)
        mask = torch.repeat_interleave(mask, topk,dim=0)
        mask = torch.repeat_interleave(mask, ngram, dim=1)
        mask = mask[:h,:w]
        return mask    

#def normalize(embs):
#    norm = np.linalg.norm(embs, axis=1)
#    norm = norm.reshape(norm.shape[0],1)
#    norm[norm==0.0] = 1.0
#    return embs / norm

def normalize_np(embs):
    return F.normalize(torch.tensor(embs), p=2, dim=1).numpy()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def mean_pooling_norm(embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    embeddings = torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return F.normalize(embeddings, p=2, dim=1)

def weighted_pooling(emb, attention_mask, window_size=8, dropoff=0.):
    if emb.dim() == 2:
        emb_exp = emb.unsqueeze(0)
    else:
        emb_exp = emb.clone()
    mask = torch.full(emb_exp.shape, dropoff).to(attention_mask.device)
    mask[:,:window_size,:] = 1.
    mask = torch.cumprod(mask, dim=1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(emb_exp.size()).float()
    mask = mask * input_mask_expanded
    #print(mask)
    emb_exp = torch.sum(emb_exp * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    #print(emb_exp)
    return F.normalize(emb_exp, p=2, dim=1)

#Encode text
def encode(inputs, args=None, model=None):
    # Compute token embeddings
    if args is not None:
        model = args.model
    with torch.no_grad():
        model_output = model(**inputs, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, inputs['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def encode_no_pool(inputs, args=None, model=None):
    # Compute token embeddings
    if args is not None:
        model = args.model
    with torch.no_grad():
        model_output = model(**inputs, return_dict=True)
    token_embeddings = model_output.last_hidden_state
    #input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    #embeddings = (token_embeddings * input_mask_expanded)

    # Normalize embeddings
    #embeddings = F.normalize(embeddings, p=2, dim=1)
    return token_embeddings




    

def pad_emb(emb_list, pad):
    max_len = max([emb.shape[0] for emb in emb_list])
    new_emb_list = [torch.concat([emb] + [pad] * (max_len-emb.shape[0]), dim=0) for emb in emb_list]
    return torch.stack(new_emb_list, dim=0)
    

def compute_sentence_perplexity(inputs, stride, args):
    max_length = args.max_length
    nlls = []
    for i in range(1, inputs["input_ids"].shape[1], stride):
        end_loc = min(i + stride, inputs["input_ids"].shape[1])
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = inputs["input_ids"][:, :end_loc].to(args.device)
        if "encoder_hidden_keys" in inputs.keys():
            encoder_indices = None
            encoder_hidden_keys = inputs.encoder_hidden_keys[:, :args.topk * (i - 1) // stride, :]
            encoder_hidden_values = inputs.encoder_hidden_values[:, :args.topk * (i - 1) // stride, :]
            encoder_attention_mask = torch.concat([inputs["encoder_attention_mask"][..., :end_loc, :end_loc], inputs["encoder_attention_mask"][..., :end_loc, inputs["input_ids"].shape[1]:inputs["input_ids"].shape[1] + args.topk * (i - 1) // stride]], dim=-1)

        elif "encoder_indices" in inputs.keys():
            encoder_hidden_keys = None
            encoder_hidden_values = None
            encoder_indices = inputs.encoder_indices[:, :args.topk * (i - 1) // stride]
            encoder_attention_mask = torch.concat([inputs["encoder_attention_mask"][..., :end_loc, :end_loc], inputs["encoder_attention_mask"][..., :end_loc, inputs["input_ids"].shape[1]:inputs["input_ids"].shape[1] + args.topk * (i - 1) // stride]], dim=-1)
        else:
            encoder_indices = None
            encoder_hidden_keys = None
            encoder_hidden_values = None
            encoder_attention_mask = None
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            args.model.eval()
            outputs = args.model(input_ids, encoder_indices=encoder_indices, encoder_hidden_keys=encoder_hidden_keys, encoder_hidden_values=encoder_hidden_values, encoder_attention_mask=encoder_attention_mask, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / end_loc)

