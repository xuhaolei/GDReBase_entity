# -*- coding:utf-8 -*-
import pandas as pd
import pickle
import numpy as np
import tqdm
import re
import csv
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
from nltk.tokenize import RegexpTokenizer

import gensim.models
import gensim.models.word2vec as w2v
import gensim.models.fasttext as fasttext

from options import dataset

MAX_SIZE = 2000
# 获取所有需要期刊编号
def get_data(outfile_path,data_type = 'disease'):
    data_type2id = {'bacterium':1,'disease':2}
    
    # step 1. 搜索期刊和实际爬取到的期刊名存在差别(存放在institution_map.xlsx)
    # 这个地方做一个字典
    df = pd.read_excel('institution_map.xlsx')
    ori2lat = {}
    for i in range(len(df)):
        if df['idxdb'][i] != df['idxdb'][i]: # nan 情况
            ori2lat[df['idxxls'][i]] = df['idxxls'][i] # 等于自身
        else:
            ori2lat[df['idxxls'][i]] = df['idxdb'][i]
    
    # step 2. 挑选出pos_paper & neg_paper
    paper_set_pos = set()
    paper_set_neg = set()
    df = pd.read_excel('译文_institution_map.xlsx')
    # 另外方法
    # df[df['label'] & data_type2id[data_type]]       -> pos
    # df[not f[df['label'] & data_type2id[data_type]] -> neg
    for i in range(len(df)):
        if df['label'][i] & data_type2id[data_type]:
            paper_set_pos.add(ori2lat[df['name'][i]])
        else:
            paper_set_neg.add(ori2lat[df['name'][i]])

    # step 3. 挑选思路，将原来的paper打乱，选择前n项pos和前n项neg
    dfpaper = pd.read_excel('paper.xlsx')
    dfpaper = dfpaper.sample(frac = 1,random_state=1).reset_index(drop=True)
    # dfpaper.filter() # 下面的方法不行 显示不可哈希
    # full_df = dfpaper[dfpaper['institution'] in paper_set_pos].limit(100)
    # full_df.append(dfpaper[dfpaper['institution'] in paper_set_neg].limit(100))
    outdic = {'title':[],'keywords':[],'abstract':[],'label':[]}
    len_pos,len_neg = 0,0
    for i in range(len(dfpaper)):
        if dfpaper['institution'][i] in paper_set_pos and len_pos<MAX_SIZE:
            outdic['title'].append(dfpaper['title'][i])
            outdic['keywords'].append(dfpaper['keywords'][i])
            outdic['abstract'].append(dfpaper['abstract'][i])
            outdic['label'].append(1)
            len_pos += 1
        elif dfpaper['institution'][i] in paper_set_neg and len_neg < MAX_SIZE:
            outdic['title'].append(dfpaper['title'][i])
            outdic['keywords'].append(dfpaper['keywords'][i])
            outdic['abstract'].append(dfpaper['abstract'][i])
            outdic['label'].append(0)
            len_neg += 1
        if len_pos == MAX_SIZE and len_neg == MAX_SIZE:
            break
    out = pd.DataFrame(outdic)
    out.to_excel(outfile_path,index = False)

# 划分训练集.开发集.测试集
def split_dataset(file_path):
    # 按照 0.8 : 0.1 : 0.1 划分训练集、开发集、测试集
    df = pd.read_excel(file_path)
    train_size,dev_size,test_size = int(0.8*len(df)),int(0.1*len(df)),int(0.1*len(df))
    df = df.sample(frac = 1).reset_index(drop=True) # 打乱
    train_df = df[:][:train_size]
    dev_df = df[:][train_size:train_size+dev_size]
    test_df = df[:][train_size+dev_size:len(df)]
    train_df.to_csv('train_'+file_path.replace('_full','').split('.')[-2]+'.csv',index = False)
    dev_df.to_csv('dev_'+file_path.replace('_full','').split('.')[-2]+'.csv',index = False)
    test_df.to_csv('test_'+file_path.replace('_full','').split('.')[-2]+'.csv',index = False)

def build_vocab(file_path,outfile_path, tokenizer, max_size=200000, min_freq=3):
    vocab_dic = {}
    with open(file_path,'r',encoding='utf-8') as f:
        lines = csv.reader(f)
        next(lines)
        cnt = 0
        for line in lines: # title 0,keywords 1,abstract 2,label 3 [0/1]
            content = line[0].strip() + line[1].strip() + line[2].strip()
            content = re.sub(r'[^\x00-\x7F]+',' ',content)
            if not content: continue
            for word in tokenizer.tokenize(content):
                word = word.lower()
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
            cnt += 1
            if cnt % 1000 == 0:
                print('%d done.'%(cnt))

    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx+1 for idx, word_count in enumerate(vocab_list)}
    # vocab_dic.update({UNK: len(vocab_dic), PAD: 0})
    with open(outfile_path,'wb') as f:
        pickle.dump(vocab_dic,f)
    return vocab_dic

def gensim_to_embeddings(wv_file, vocab_file,outfile=None):
    model = gensim.models.Word2Vec.load(wv_file)
    wv = model.wv
    #free up memory
    del model

    with open(vocab_file,'rb') as f:
        vocab = pickle.load(f)

    # ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
    ind2w = {vocab[key]:key for key in vocab}
    W, words = build_matrix(ind2w, wv)

    if outfile is None:
        outfile = wv_file.replace('.w2v', '.embed')

    #smash that save button
    save_embeddings(W, words, outfile)


def word_embeddings(modelname,notes_file,tokenizer, embedding_size, min_count, n_iter):

    sentences = ProcessedIter(notes_file,tokenizer)

    model = w2v.Word2Vec(vector_size=embedding_size, min_count=min_count, workers=4, epochs=n_iter)
    print("building word2vec vocab on %s..." % (notes_file))

    model.build_vocab(sentences)
    print("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    out_file = '/'.join(notes_file.split('/')[:-1] + [modelname])
    print("writing embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file

def build_matrix(ind2w, wv):
    """
        Go through vocab in order. Find vocab word in wv.index2word, then call wv.word_vec(wv.index2word[i]).
        Put results into one big matrix.
        Note: ind2w starts at 1 (saving 0 for the pad character), but gensim word vectors starts at 0
    """
    W = np.zeros((len(ind2w)+1, len(wv.word_vec(wv.index_to_key[0])) ))
    words = ["**PAD**"]
    W[0][:] = np.zeros(len(wv.word_vec(wv.index_to_key[0])))
    for idx, word in ind2w.items():
        if idx >= W.shape[0]:
            break
        W[idx][:] = wv.word_vec(word)
        words.append(word)
    return W, words

def save_embeddings(W, words, outfile):
    with open(outfile, 'w') as o:
        #pad token already included
        for i in range(len(words)):
            line = [words[i]]
            line.extend([str(d) for d in W[i]])
            o.write(" ".join(line) + "\n")

class ProcessedIter(object):

    def __init__(self, filename,tokenizer):
        self.filename = filename
        self.tokenizer = tokenizer
    def __iter__(self):
        with open(self.filename,'r',encoding='utf-8') as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                content = ''.join([row[i].strip() for i in range(3)])
                content = re.sub(r'[^\x00-\x7F]+',' ',content)
                if content == '': continue
                content = self.tokenizer.tokenize(content)
                content = [c.lower() for c in content]
                yield content
def load_embeddings(embed_file):
    #also normalizes the embeddings
    W = []
    with open(embed_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
        #UNK embedding, gaussian randomly initialized
        print("adding unk embedding")
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    W = np.array(W)
    return W

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
def micro_precision(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    if flat_pred.sum(axis=0) == 0:
        return 0.0
    return intersect_size(flat_true, flat_pred, 0) / flat_pred.sum(axis=0)


def micro_recall(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    return intersect_size(flat_true, flat_pred, 0) / flat_true.sum(axis=0)


def micro_f1(true_labels, pred_labels):
    prec = micro_precision(true_labels, pred_labels)
    rec = micro_recall(true_labels, pred_labels)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


def micro_accuracy(true_labels, pred_labels):
    flat_true = true_labels.ravel()    # 将多维数组转换为1维数组
    flat_pred = pred_labels.ravel()
    return intersect_size(flat_true, flat_pred, 0) / union_size(flat_true, flat_pred, 0)


def macro_precision(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (pred_labels.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (true_labels.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(true_labels, pred_labels):
    prec = macro_precision(true_labels, pred_labels)
    rec = macro_recall(true_labels, pred_labels)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


def union_size(x, y, axis):       # 或逻辑后加和
    return np.logical_or(x, y).sum(axis=axis).astype(float)


def intersect_size(x, y, axis):   # 与逻辑后加和
    return np.logical_and(x, y).sum(axis=axis).astype(float)

def evaluate(model, data_iter, criterions, device):
    model.eval()
    loss_total = 0
    test_acc = 0
    i = 0
    test_P_macro, test_R_macro, test_f1_macro = 0, 0, 0
    test_P_micro, test_R_micro, test_f1_micro = 0, 0, 0
    all_labels = []
    all_pred_labels = []
    with torch.no_grad():

        for sentences,labels in data_iter:
            # step 5.1 将数据放到设备上(cpu or gpu)
            sentences,labels = sentences.to(device),labels.to(device)

            outputs = model(sentences)

            loss = criterions(outputs, labels.float())
            loss_total += loss

            labels = labels.data.cpu()

            pred_prob = outputs.data
            pred_labels_1 = pred_prob.cpu().numpy().copy()
            pred_labels = np.zeros_like(pred_labels_1)
            pred_labels[pred_labels_1>0.5] = 1
            
            all_pred_labels.append(pred_labels)
            all_labels.append(labels.numpy())
        
        all_pred_labels = np.concatenate(all_pred_labels,axis = 0)
        all_labels = np.concatenate(all_labels,axis = 0)

    macro = macro_f1(all_labels, all_pred_labels)
    micro = micro_f1(all_labels, all_pred_labels)
    return macro, micro, loss_total / len(data_iter)


if __name__ == "__main__":
    # 从原始paper.xlsx中获取数据，放到disease_full.xlsx中
    assert dataset in ['disease','bacterium']
    get_data('%s_full.xlsx'%dataset,dataset)
    # 划分训练集，开发集，测试集 -> train_disease.csv,dev_disease.csv,test_disease.csv
    split_dataset('%s_full.xlsx'%dataset)

    # 构件词表 -> vocab.pkl
    tokenizer = RegexpTokenizer(r'\w+')
    build_vocab('train_%s.csv'%dataset,'vocab_%s.pkl'%dataset, tokenizer, 25000, 3)

    # 得到预训练词向量，最终得到processed.embed
    w2v_file = word_embeddings('processed-%s.w2v'%dataset,'train_%s.csv'%dataset,tokenizer, 100, 0, 10)
    gensim_to_embeddings('processed-%s.w2v'%dataset, 'vocab_%s.pkl'%dataset)

