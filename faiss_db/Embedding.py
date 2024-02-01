import torch
import torch.nn as nn

class ContextEmbed(nn.Module):
    def __init__(self, embedding_path=None, size=None, dim=None):
        super().__init__()
        self.embed = None
        if embedding_path is not None:
            self.embed = torch.load(embedding_path)
        elif size is not None and dim is not None:
            self.embed = nn.Embedding(size,dim)
        self.embed.weight.requires_grad = False
    
    def from_pretrained(self, embedding_path):
        self.embed = torch.load(embedding_path)
        self.embed.weight.requires_grad = False

    def forward(self, x):
        assert self.embed is not None
        return self.embed(x)
