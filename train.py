"""
author: Mazharul Islam Leon
created_at: 2021-12-15

This is trainning file of this image classifier

"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from data import train_dataloader,train_datasets
import cfg
from utils import adjust_learning_rate_cosine, adjust_learning_rate_step


## stored the model with weights and biases
save_folder = cfg.SAVE_FOLDER + cfg.model_name
os.makedirs(save_folder, exist_ok=True)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    return model


def main():
    #####build the network model
    if not cfg.RESUME_EPOCH:
        print('****** Training {} ****** '.format(cfg.model_name))
        print('****** loading the Imagenet pretrained weights ****** ')
        if not cfg.model_name.startswith('efficientnet'):
            model = cfg.MODEL_NAMES[cfg.model_name](num_classes=cfg.NUM_CLASSES)
            # #冻结前边一部分层不训练
            ct = 0
            for child in model.children():
                ct += 1
                # print(child)
                if ct < 8:
                    print(child)
                    for param in child.parameters():
                        param.requires_grad = False
        else:
            model = cfg.MODEL_NAMES[cfg.model_name](cfg.model_name,num_classes=cfg.NUM_CLASSES)
            # print(model)
            c = 0
            for name, p in model.named_parameters():
                c += 1
                # print(name)
                if c >=700:
                    break
                p.requires_grad = False

        # print(model)
    if cfg.RESUME_EPOCH:
        print(' ******* Resume training from {}  epoch {} *********'.format(cfg.model_name, cfg.RESUME_EPOCH))
        model = load_checkpoint(os.path.join(save_folder, 'epoch_{}.pth'.format(cfg.RESUME_EPOCH)))



    # GPU number check from config file
    if cfg.GPUS>1:
        print('****** using multiple gpus to training ********')
        model = nn.DataParallel(model,device_ids=list(range(cfg.GPUS)))
    else:
        print('****** using single gpu to training ********')
    print("...... Initialize the network done!!! .......")

    ### check cuda availability and transfer model to cuda
    if torch.cuda.is_available():
        model.cuda()


    ## load the Adam optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR)
    # optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    optimizer = optim.SGD(model.parameters(), lr=cfg.LR,
                        momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = LabelSmoothSoftmaxCE()
    # criterion = LabelSmoothingCrossEntropy()

    lr = cfg.LR

    batch_size = cfg.BATCH_SIZE

    # epoch and batch size
    max_batch = len(train_datasets)//batch_size
    epoch_size = len(train_datasets) // batch_size
    ## max epoch
    max_iter = cfg.MAX_EPOCH * epoch_size

    start_iter = cfg.RESUME_EPOCH * epoch_size

    epoch = cfg.RESUME_EPOCH

    # cosine
    warmup_epoch=5
    warmup_steps = warmup_epoch * epoch_size
    global_step = 0

    # step 
    stepvalues = (10 * epoch_size, 20 * epoch_size, 30 * epoch_size)
    step_index = 0

    model.train()
    for iteration in range(start_iter, max_iter):
        global_step += 1

        ##
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(train_dataloader)
            loss = 0
            epoch += 1
            ###保存模型
            if epoch % 5 == 0 and epoch > 0:
                if cfg.GPUS > 1:
                    checkpoint = {'model': model.module,
                                'model_state_dict': model.module.state_dict(),
                                # 'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch}
                    torch.save(checkpoint, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))
                else:
                    checkpoint = {'model': model,
                                'model_state_dict': model.state_dict(),
                                # 'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch}
                    torch.save(checkpoint, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))

        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate_step(optimizer, cfg.LR, 0.1, epoch, step_index, iteration, epoch_size)

        ## for using learing cosine adjust
        # lr = adjust_learning_rate_cosine(optimizer, global_step=global_step,
        #                           learning_rate_base=cfg.LR,
        #                           total_steps=max_iter,
        #                           warmup_steps=warmup_steps)


        ## image label
        # try:
        images, labels = next(batch_iterator)
        # except:
        #     continue

        # Check if cuda is available or not
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

        out = model(images)
        loss = criterion(out, torch.tensor(labels, dtype=torch.long))

        optimizer.zero_grad()  
        loss.backward()  # loss function backward
        optimizer.step()  # update weights

        prediction = torch.max(out, 1)[1]
        train_correct = (prediction == labels).sum()
        
        # print(train_correct.type())
        train_acc = (train_correct.float()) / batch_size

        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                + '|| Totel iter ' + repr(iteration) + ' || Loss: %.6f||' % (loss.item()) + 'ACC: %.3f ||' %(train_acc * 100) + 'LR: %.8f' % (lr))
        
    print("******")



if __name__ == '__main__':
    main()