import os
# import glob
from pathlib import  Path
import sys 
#sys.path.append(".")  # add the path of the parent directory") 
sys.path.append(os.path.abspath(".."))
import cfg
import random



if __name__ == '__main__':
    
    traindata_path = cfg.BASE / "train"
    # print(traindata_path)
    # Join BASE and test path
    test_path = cfg.BASE / "test"
    labels = os.listdir(str(traindata_path))
    # print(labels)
    # to join the validate path
    #valdata_path = cfg.BASE / "val"
    
    ##
    txtpath = cfg.BASE
    # print(labels)
    for index, label in enumerate(labels):
        lablel_path = traindata_path / label
        #imglist = glob.glob(os.path.join(traindata_path,label, '*.jpg'))
        imglist = list(lablel_path.glob('*.*'))
        random.shuffle(imglist)
        print(len(imglist))
        trainlist = imglist[:int(0.8*len(imglist))]
        vallist = imglist[(int(0.8*len(imglist))+1):]
        with open(txtpath / 'train.txt', 'a')as f:
            for img in trainlist:
                #print(img + ' ' + str(index))
                f.write(str(img) + ' ' + str(index))
                f.write('\n')

        with open(txtpath / 'val.txt', 'a')as f:
            for img in vallist:
                # print(img + ' ' + str(index))
                f.write(str(img) + ' ' + str(index))
                f.write('\n')


    #imglist = glob.glob(os.path.join(valdata_path, '*.jpg'))
    imglist = imglist = list(test_path.glob('**/*.*'))
    print("Test: ",len(imglist))
    with open(txtpath / 'test.txt', 'a')as f:
        for img in imglist:
            f.write(str(img))
            f.write('\n')