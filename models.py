# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_embeddings

'''TextCNN'''

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        config.filter_sizes = config.filter_sizes.split(',')
        config.filter_sizes = [int(item) for item in config.filter_sizes]
        assert config.embed_file is not None # 断言
        W = torch.Tensor(load_embeddings(config.embed_file))
        self.embedding = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embedding.weight.data = W.clone()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, W.size()[1])) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), 256)

        self.outs = nn.Linear(256,config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):        
        out = self.embedding(x)
        
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = torch.relu(self.fc(out))

        out = F.sigmoid(self.outs(out))

        return out

