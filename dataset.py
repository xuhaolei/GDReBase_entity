# -*- coding:utf-8 -*-
import numpy as np
import csv
import re
import pickle

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import nltk
from nltk.tokenize import RegexpTokenizer

class PaperDataset(Dataset):
    """docstring for PaperDataset"""
    def __init__(self,file,args):
        super(PaperDataset, self).__init__()
        with open(args.vocab_path,'rb') as f:
            vocab = pickle.load(f)
        tokenizer = RegexpTokenizer(r'\w+')
        self.data_list = []
        with open(file,'r',encoding='utf-8') as f:
            lines = csv.reader(f)
            next(lines)
            for line in lines:
                content = ''.join([line[i].strip() for i in range(3)])
                content = re.sub(r'[^\x00-\x7F]+',' ',content)
                words = tokenizer.tokenize(content)
                words = [word.lower() for word in words]
                token_ids = [vocab[word] if word in vocab else len(vocab)+1 for word in words]
                if len(token_ids)>args.pad_size:
                    token_ids = token_ids[:args.pad_size]
                else:
                    token_ids = token_ids + [0] * (args.pad_size - len(token_ids))
                self.data_list.append((torch.LongTensor(token_ids),torch.LongTensor([int(line[3])]))) 

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item][0],self.data_list[item][1]
