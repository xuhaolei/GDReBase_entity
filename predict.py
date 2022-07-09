# -*- coding:utf-8 -*-
import time
import os
import pickle
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import nltk
from nltk.tokenize import RegexpTokenizer

from options import args

class Predict(object):
	"""docstring for Predict"""
	def __init__(self, args):
		super(Predict, self).__init__()
		# 原来模型的一些参数
		self.args = args
		self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda:%d'%args.gpu)
		# 词表地址
		self.bacterium_vocab_path = 'vocab_bacterium.pkl'
		self.disease_vocab_path = 'vocab_disease.pkl'
		# 模型地址
		self.bacterium_model_path = 'TextCNN-bacterium.pth'
		self.disease_model_path = 'TextCNN-disease.pth'
		# 分词
		self.tokenizer = RegexpTokenizer(r'\w+')

		self.bacterium_model = torch.load(self.bacterium_model_path).to(self.device)
		self.disease_model = torch.load(self.disease_model_path).to(self.device)
		with open(self.bacterium_vocab_path,'rb') as f:
			self.bacterium_vocab = pickle.load(f)
		with open(self.disease_vocab_path,'rb') as f:
			self.disease_vocab = pickle.load(f)

	def predict(self,content):
		content = re.sub(r'[^\x00-\x7F]+',' ',content)
		words = self.tokenizer.tokenize(content)
		words = [word.lower() for word in words]
		
		# 此时分成疾病细菌两部分
		token_bacterium_ids = [self.bacterium_vocab[word] if word in self.bacterium_vocab\
								 else len(self.bacterium_vocab)+1 for word in words]
		token_disease_ids = [self.disease_vocab[word] if word in self.disease_vocab\
								 else len(self.disease_vocab)+1 for word in words]
		# 分别进行padding
		if len(token_bacterium_ids)>self.args.pad_size:
			token_bacterium_ids = token_bacterium_ids[:self.args.pad_size]
		else:
			token_bacterium_ids = token_bacterium_ids + [0] * (self.args.pad_size - len(token_bacterium_ids))
		if len(token_disease_ids)>self.args.pad_size:
			token_disease_ids = token_disease_ids[:self.args.pad_size]
		else:
			token_disease_ids = token_disease_ids + [0] * (self.args.pad_size - len(token_disease_ids))

		bacterium_output = self.bacterium_model(torch.LongTensor(token_bacterium_ids).unsqueeze(0).to(self.device))
		disease_output = self.disease_model(torch.LongTensor(token_disease_ids).unsqueeze(0).to(self.device))
		bacterium_output = bacterium_output.data.cpu().numpy().copy()[0,0]
		disease_output = disease_output.data.cpu().numpy().copy()[0,0]
		# print(bacterium_output)
		# print(disease_output)
		if bacterium_output>0.2 and disease_output>0.2:
			return True
		else:
			return False

# predict = Predict(args)
# content = """
# Spleen tyrosine kinase (SYK) signaling pathway regulates 
# critical processes in innate immunity, but its role in parenchymal cells remains 
# elusive in chronic liver diseases. We investigate the relative contribution of 
# SYK and its substrate c-Abl Src homology 3 domain-binding protein-2 (3BP2) in 
# both myeloid cells and hepatocytes in the onset of metabolic steatohepatitis.
# Hepatic SYK-3BP2 pathway was evaluated in mouse models of 
# metabolic-associated fatty liver diseases (MAFLD) and in obese patients with 
# biopsy-proven MAFLD (n = 33). Its role in liver complications was evaluated in 
# Sh3bp2 KO and myeloid-specific Syk KO mice challenged with methionine and 
# choline deﬁcient diet and in homozygous Sh3bp2KI/KI mice with and without SYK 
# expression in myeloid cells.
# Here we report that hepatic expression of 3BP2 and SYK correlated with 
# metabolic steatohepatitis severity in mice. 3BP2 deficiency and SYK deletion in 
# myeloid cells mediated the same protective effects on liver inflammation, 
# injury, and fibrosis priming upon diet-induced steatohepatitis. In primary 
# hepatocytes, the targeting of 3BP2 or SYK strongly decreased the 
# lipopolysaccharide-mediated inflammatory mediator expression and 3BP2-regulated 
# SYK expression. In homozygous Sh3bp2KI/KI mice, the chronic inflammation 
# mediated by the proteasome-resistant 3BP2 mutant promoted severe hepatitis and 
# liver fibrosis with augmented liver SYK expression. In these mice, the deletion 
# of SYK in myeloid cells was sufficient to prevent these liver lesions. The 
# hepatic expression of SYK is also up-regulated with metabolic steatohepatitis 
# and correlates with liver macrophages in biopsy-proven MAFLD patients.
# Collectively, these data suggest an important role for the SYK-3BP2 
# pathway in the pathogenesis of chronic liver inflammatory diseases and highlight 
# its targeting in hepatocytes and myeloid cells as a potential strategy to treat 
# metabolic steatohepatitis.
# """

# predict.predict(content)