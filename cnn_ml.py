# author: Mazharul Islam Leon

import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.externals import joblib
import pandas as pd

import cfg

import os
from PIL import Image
from transform import get_test_transform
from tqdm import tqdm
import torch.nn as nn
import pickle
import numpy as np

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # print(checkpoint)
    model = checkpoint['model']  # model checkpoint
    # print(model)
    model.load_state_dict(checkpoint['model_state_dict'])  # load the model state dict
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


NB_features = 2816
def save_feature(model, feature_path, label_path):
    '''
    
    '''
    model = load_checkpoint(model)
    # print(model)
    print('..... Finished loading model! ......')
    ## Check the cuda availability
    if torch.cuda.is_available():
        model.cuda()
    ## 
    nb_features = NB_features
    features = np.empty((len(imgs), nb_features))
    labels = []
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip().split(' ')[0]
        label = imgs[i].strip().split(' ')[1]
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        # print(type(img))
        img = get_test_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)

        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            out = model.extract_features(img)
            # print(out.size())
            out2 = nn.AdaptiveAvgPool2d(1)(out)
            feature = out2.view(out.size(1), -1).squeeze(1)
            # print(out3.size())
            # print(out2.size())
        features[i, :] = feature.cpu().numpy()
        labels.append(label)

    pickle.dump(features, open(feature_path, 'wb'))
    pickle.dump(labels, open(label_path, 'wb'))
    print('CNN features obtained and saved.')



def classifier_training(feature_path, label_path, save_path):
    '''
    
    '''
    print('Pre-extracted features and labels found. Loading them ...')
    features = pickle.load(open(feature_path, 'rb'))
    labels = pickle.load(open(label_path, 'rb'))
    classifier = SVC(C=0.5)
    # classifier = MLPClassifier()
    # classifier = RandomForestClassifier(n_jobs=4, criterion='entropy', n_estimators=70, min_samples_split=5)
    # classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    # classifier = ExtraTreesClassifier(n_jobs=4,  n_estimators=100, criterion='gini', min_samples_split=10,
    #                        max_features=50, max_depth=40, min_samples_leaf=4)
    # classifier = GaussianNB()
    print(".... Start fitting this classifier ....")
    classifier.fit(features, labels)
    print("... Training process is down. Save the model ....")
    joblib.dump(classifier, save_path)
    print("... model is saved ...")



def classifier_pred(model_path, feature, id):
    '''
    
    '''
    features = pickle.load(open(feature, 'rb'))
    ids = pickle.load(open(id, 'rb'))
    print("... loading the model ...")
    classifier = joblib.load(model_path)
    print("... load model done and start predicting ...")
    predict = classifier.predict(features)
    # print(type(predict))
    # print(predict.shape)
    # print(ids)
    prediction = predict.tolist()
    submission = pd.DataFrame({"ID": ids, "Label": prediction})
    submission.to_csv('../' + 'svm_submission.csv', index=False, header=False)




if __name__ == "__main__":
    # 
    feature_path = '../features/'
    os.makedirs(feature_path, exist_ok=True)

#############################################################
    ####
    with open(cfg.TRAIN_LABEL_DIR, 'r')as f:
        imgs = f.readlines()
    train_feature_path = feature_path + 'psdufeature.pkl'
    train_label_path = feature_path + 'psdulabel.pkl'
    cnn_model = cfg.TRAINED_MODEL
    save_feature(cnn_model, train_feature_path, train_label_path)
    #
    ## #????????????????????????
    train_feature_path = feature_path + 'psdufeature.pkl'
    train_label_path = feature_path + 'psdulabel.pkl'
    save_path = feature_path + 'psdusvm.m'
    classifier_training(train_feature_path, train_label_path, save_path)

######################################################################

    with open(cfg.VAL_LABEL_DIR, 'r')as f:
        imgs = f.readlines()
    test_feature_path = feature_path + 'testfeature.pkl'
    test_id_path = feature_path + 'testid.pkl'
    cnn_model = cfg.TRAINED_MODEL
    save_feature(cnn_model, test_feature_path, test_id_path)


    ## #????????????
    test_feature_path = feature_path + 'testfeature.pkl'
    test_id_path = feature_path + 'testid.pkl'
    save_path = feature_path + 'svm.m'
    classifier_pred(save_path, test_feature_path, test_id_path)