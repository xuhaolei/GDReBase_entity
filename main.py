# -*- coding:utf-8 -*-
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from options import args
from dataset import PaperDataset
from models import TextCNN
from utils import init_network,evaluate

if __name__ == '__main__':
    device = torch.device('cpu' if args.gpu==-1 else 'cuda:%d'%(args.gpu))
    print('loadind dataset...')
    # step 1. 加载数据集
    train_set,dev_set,test_set = PaperDataset(args.train_path,args),\
                                 PaperDataset(args.dev_path,args),\
                                 PaperDataset(args.test_path,args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                             num_workers=1, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,
                             num_workers=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=1, shuffle=True)
    
    # step 2. 加载网络模型，并进行初始化，要是有上一轮训练的模型就接着训练
    model = TextCNN(args).to(device)
    if os.path.exists(args.save_path): # 接着上一次训练
        model.load_state_dict(torch.load(args.save_path))
    else:
        init_network(model)

    # step 3. 定义优化器(adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # step 4. 定义损失函数(BCE_Loss_function)
    criterions = nn.BCELoss()
    print('training...')

    # step 5. 定义flag
    _, micro, _ = evaluate(model, dev_loader, criterions,device) # 防止之前训练的模型被坏的模型冲掉
    best_f1 = micro[2] # 当前最好f1
    if best_f1 != best_f1: # nan
        best_f1 = 0
    improve = 0        # 没达到最好f1的轮数
    print('best_f1:%.2f'%best_f1)
    # step 6. 训练模型
    for epoch in range(args.num_epochs):
        if args.testonly: continue
        print('Epoch [{}/{}]'.format(epoch + 1, args.num_epochs))
        now = time.time()
        for sentences,labels in train_loader:
            # step 6.1 将数据放到设备上(cpu or gpu)
            sentences,labels = sentences.to(device),labels.to(device)
            
            # step 6.2 正向传播，从模型得到结果output
            model.train()
            model.zero_grad()
            outputs = model(sentences)

            # step 6.3 反向传播，计算损失，梯度求导
            loss = criterions(outputs, labels.float())
            loss.backward()
            optimizer.step()

        # 6.4 每轮后用进行评估
        # train_macro, train_micro, train_loss = evaluate(model, train_loader, criterions,device)
        dev_macro, dev_micro, dev_loss = evaluate(model, dev_loader, criterions,device)
        msg = 'epoch: {0:>6}, Dev Loss: {1:>5.7}, Dev P(macro): {2:>6.2%}, Dev R(macro): {3:>6.2%}, Dev F1(macro): {4:>6.2%}\n' \
               '              Dev P(micro): {5:>6.2%}, Dev R(micro): {6:>6.2%}, Dev F1(micro): {7:>6.2%}' \
               '              Time: {8}'
        print(msg.format(epoch+1, dev_loss, dev_macro[0], dev_macro[1], dev_macro[2],
                                 dev_micro[0], dev_micro[1], dev_micro[2], time.time() - now))
        if dev_micro[2] > best_f1:
            improve = 0
            # 保存模型
            torch.save(model.state_dict(), args.save_path)
        else:
            improve += 1
            # 早停机制
            if improve > args.patience:
                break

    # step 7. 利用最好模型得到测试集结果
    print('testing...')
    model.load_state_dict(torch.load(args.save_path))
    torch.save(model, args.save_path) # 保存整个模型
    test_macro, test_micro, test_loss = evaluate(model, test_loader, criterions,device)

    msg = 'Test Loss : {0:>5.7},Test P(macro) : {1:>6.2%},Test R(macro) : {2:>6.2%},Test F1(macro) : {3:>6.2%}\n'\
          '               Test P(micro) : {4:>6.2%},Test R(micro) : {5:>6.2%},Test F1(micro) : {6:>6.2%},\n'
    print(msg.format(test_loss,test_macro[0],test_macro[1],test_macro[2],test_micro[0],test_micro[1],test_macro[2]))
