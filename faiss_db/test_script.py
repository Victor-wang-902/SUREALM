from transformers import AutoModel, AutoConfig, AutoTokenizer
import models as m
import sys
''' test 1
sys.stdout = open('file', 'w')
config = AutoConfig.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
model = m.RetrievalModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
model2 = m.RetrievalModel(config)
model3 = model2.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
with open("res.txt", "w") as f:
    for p,q,r in zip(model.named_parameters(), model2.named_parameters(), model3.named_parameters()):
        print(p)
        print(q)
        print(r)
sys.stdout.close()
'''
''' test 2
sys.stdout = open('file3', 'w')
model = m.RetrievalModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model.is_retrieval("dummy_pre.pt", "dummy_post.pt")
print(model)
sys.stdout.close()
'''
'''#test 3
sys.stdout = open('file3', 'w')
config = AutoConfig.from_pretrained("./multi-qa-MiniLM-L6-cos-v1")
config.is_decoder = True
model = m.RetrievalModel(config)
print(model.config.is_decoder)
model = m.RetrievalModel.from_pretrained("./multi-qa-MiniLM-L6-cos-v1")
model.is_retrieval("dummy_pre.pt", "dummy_post.pt")
print(model)
print(model.config.is_decoder)
print(model.encoder.layer[0].is_decoder)
sys.stdout.close()
'''
''' test 4
sys.stdout = open('file4', 'w')
config = AutoConfig.from_pretrained("./multi-qa-MiniLM-L6-cos-v1")#, local_files_only=True)
model = m.RetrievalModel(config)

print(model.config.add_cross_attention)
print(model.config.is_decoder)
print(model)
model = m.RetrievalModel.from_pretrained("./multi-qa-MiniLM-L6-cos-v1")#, ignore_mismatched_sizes=True, local_files_only=True)
model.is_retrieval("dummy_pre.pt", "dummy_post.pt")
print(model)
print(model.config.add_cross_attention)
sys.stdout.close()
'''
''' test 5
sys.stdout = open('file5', 'w')
model = m.RetrievalModel.from_pretrained("./modified_multi-qa-MiniLM-L6-cos-v1")#, ignore_mismatched_sizes=True, local_files_only=True)
model.is_retrieval("dummy_pre.pt", "dummy_post.pt")
model.init_cross_attention_weights()
model2 = m.RetrievalModel.from_pretrained("./modified_multi-qa-MiniLM-L6-cos-v1")
model2.is_retrieval("dummy_pre.pt", "dummy_post.pt")
for p,q in zip(model.named_parameters(), model2.named_parameters()):
        print(p)
        print(q)
print(model)
print(model.config.add_cross_attention)
sys.stdout.close()
'''
''' test 6
sys.stdout = open('file6', 'w')
model = m.RetrievalModel.from_pretrained("./modified_multi-qa-MiniLM-L6-cos-v1")#, ignore_mismatched_sizes=True, local_files_only=True)
model.is_retrieval("dummy_pre.pt", "dummy_post.pt")
model.init_cross_attention_weights()
for p in model.named_parameters():
        print(p)
print(model)
print(model.config.add_cross_attention)
sys.stdout.close()
''' 
'''test7
sys.stdout = open('file7', 'w')
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")#, ignore_mismatched_sizes=True, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

sentence = ["I am sad", "I am sad"]

sentence2 = ["[CLS] I am sad", "[CLS] I am sad [SEP]", "I am sad[SEP]"]

with torch.no_grad():
        encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')

        encoded_input2 = tokenizer(sentence2, padding=True, add_special_tokens=False, truncation=True, return_tensors='pt')
        print(encoded_input)
        print(encoded_input2)
        output = model(**encoded_input)
        output2 = model(**encoded_input2)

        emb = mean_pooling(output, encoded_input["attention_mask"]).numpy()
        emb2 = mean_pooling(output2, encoded_input2["attention_mask"]).numpy()
        print(emb.shape)
        print(cosine_similarity([emb[0],emb[1]]))
        print(cosine_similarity([emb2[0],emb2[1],emb2[2]]))

sys.stdout.close()
'''
