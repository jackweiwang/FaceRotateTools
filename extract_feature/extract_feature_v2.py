# Helper function for extracting features from pre-trained models

import sys, os
dirname, filename = os.path.split(os.path.realpath(__file__))
sys.path.append(dirname)


import torch
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import model_irse

from sklearn.decomposition import PCA
import sklearn

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def calculate_roc(embeddings11, embeddings12, threshold = 1.3,  pca = 1):

    embeddings1 = embeddings11.numpy()

    acc_train = np.zeros((len(embeddings12)))
    dist = np.full( len(embeddings12), 0)

    for i in range(len(embeddings12)):

        embeddings2 = embeddings12[i].numpy()
        if pca == 0:
            diff = np.subtract(embeddings1, embeddings2)
            dist = np.sum(np.square(diff), 1)

            # print('train_set', train_set)
            # print('test_set', test_set)
        if pca > 0:

            embed1_train = embeddings1[0, :].reshape(1,-1)
            embed2_train = embeddings2[0, :].reshape(1,-1)

            _embed_train = np.concatenate((embed1_train, embed2_train), axis = 0)
            pca_model = PCA(n_components = pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embed1_train)
            embed2 = pca_model.transform(embed2_train)

            diff = np.subtract(embed1, embed2)

            dist[i] = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold

    acc_train = np.less(dist, threshold)

    best_threshold_index = np.argmin(acc_train)
    print('best_threshold_index', best_threshold_index)

    return best_threshold_index, dist[best_threshold_index]

def compare(inA,inB):#inA,inB是列向量
    f11 = 0
    f22 = 0
    f12 = 0
    NinA = inA.numpy()
    NinB = inB.numpy()
    #print(NinA.shape)
    for i in range(512):
     
        temp1 = NinA[0, i]

        f11 += temp1 * temp1

        temp2 = NinB[0, i]

        f22 += temp2 * temp2

        f12 += temp1 * temp2

    score = f12 / math.sqrt(f11 * f22)

    #print('score = ', score)
    if score < -1.0:
        score = -1
    elif score > 1.0:
        score = -1
    elif score < 0.0:
        score = 0
    return score



def initHandel(model_root):
    size_mo = [112, 112]

    backbone = model_irse.IR_152(size_mo)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root, map_location=lambda storage, loc: storage))

    backbone.to(device)

    # extract features
    backbone.eval() # set to evaluation mode

    return backbone


def extract_feature(img, backbone, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = True):
    # pre-requisites
    #assert(os.path.exists(img_root))
    #print('Testing Data Root:', img_root)

    # load image
    #img = cv2.imread(img_root)

    # resize image to [128, 128]
    resized = cv2.resize(img, (128, 128))


    # center crop image
    a=int((128-112)/2) # x start
    b=int((128-112)/2+112) # x end
    c=int((128-112)/2) # y start
    d=int((128-112)/2+112) # y end
    ccropped = resized[a:b, c:d] # center crop the image
    ccropped = ccropped[...,::-1] # BGR to RGB

    # flip image horizontally
    flipped = cv2.flip(ccropped, 1)

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype = np.float32)
    flipped = (flipped - 127.5) / 128.0
    flipped = torch.from_numpy(flipped)


   
    with torch.no_grad():
        if tta:
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(ccropped.to(device)).cpu())
            
    #np.save("features.npy", features) 
#     features = np.load("features.npy")

    return features
