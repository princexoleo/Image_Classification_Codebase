"""
author: Mazharul Islam Leon
created_at: 2021-12-15

This is predict file of this image classifier
"""
import torch
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter
import cfg
from data import tta_test_transform, get_test_transform

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # save into the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # save the weight and bias of the model
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def predict(model):
    # load the model from checkpoint
    model = load_checkpoint(model)
    print('..... Finished loading model! ......')
    ## check cuda availability
    if torch.cuda.is_available():
        model.cuda()
    pred_list, _id = [], []
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip()
        # print(img_path)
        _id.append(os.path.basename(img_path).split('.')[0])
        img = Image.open(img_path).convert('RGB')
        # print(type(img))
        img = get_test_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)

        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            out = model(img)
        prediction = torch.argmax(out, dim=1).cpu().item()
        pred_list.append(prediction)
    return _id, pred_list


def tta_predict(model):
    # load the model from checkpoint
    model = load_checkpoint(model)
    print('..... Finished loading model! ......')
    ## check cuda availability
    if torch.cuda.is_available():
        model.cuda()
    pred_list, _id = [], []
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip()
        # print(img_path)
        _id.append(int(os.path.basename(img_path).split('.')[0]))
        img1 = Image.open(img_path).convert('RGB')
        # print(type(img))
        pred = []
        for i in range(8):
            img = tta_test_transform(size=cfg.INPUT_SIZE)(img1).unsqueeze(0)

            if torch.cuda.is_available():
                img = img.cuda()
            with torch.no_grad():
                out = model(img)
            prediction = torch.argmax(out, dim=1).cpu().item()
            pred.append(prediction)
        res = Counter(pred).most_common(1)[0][0]
        pred_list.append(res)
    return _id, pred_list


if __name__ == "__main__":

    trained_model = cfg.TRAINED_MODEL
    model_name = cfg.model_name
    with open(cfg.TEST_LABEL_DIR,  'r')as f:
        imgs = f.readlines()

    # _id, pred_list = tta_predict(trained_model)
    _id, pred_list = predict(trained_model)

    submission = pd.DataFrame({"ID": _id, "Label": pred_list})
    submission.to_csv(cfg.BASE + '{}_submission.csv'
                      .format(model_name), index=False, header=False)


