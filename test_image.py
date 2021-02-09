import argparse
import time
import warnings
import numpy as np
import torch
import math
import torchvision
from torchvision import transforms
import cv2
import os
from dectect import AntiSpoofPredict

from pfld.pfld import PFLDInference, AuxiliaryNet

from extract_feature import model_irse
from extract_feature import landmarks_alignment
from extract_feature import extract_feature_v2
#import dlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_process(args, inputname, transform, plfd_backbone, handle):

    img = cv2.imread(inputname)
    width = img.shape[1]
    height = img.shape[0]

    model_test = AntiSpoofPredict(device)
    image_bbox = model_test.get_bbox(img)
    if image_bbox == None:

        return
    x1 = image_bbox[0]
    y1 = image_bbox[1]
    x2 = image_bbox[0] + image_bbox[2]
    y2 = image_bbox[1] + image_bbox[3]
    w = x2 - x1
    h = y2 - y1

    size = int(max([w, h]))
    cx = x1 + w/2
    cy = y1 + h/2
    x1 = cx - size/2
    x2 = x1 + size
    y1 = cy - size/2
    y2 = y1 + size

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)

    cropped1 = img[int(y1):int(y2), int(x1):int(x2)]

    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
        dx = int(dx)
        dy = int(dy)
        edx = int(edx)
        edy = int(edy)
        cropped1 = cv2.copyMakeBorder(cropped1, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
         

    cropped = cv2.resize(cropped1, (112, 112))
    
    input = cropped.copy()
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    input = transform(input).unsqueeze(0).to(device)
    _, landmarks = plfd_backbone(input)
    pre_landmark = landmarks[0]
    pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]


    src_list = pre_landmark.tolist()
    RotateMatrix = landmarks_alignment.warp_im(src_list)
    rev_aligned = landmarks_alignment.alignlandmarks(cropped, RotateMatrix)
    #cv2.imwrite('test.jpg', rev_aligned)
    
    feature = extract_feature_v2.extract_feature(rev_aligned, handle)

    return feature

def run(args, A, B, transform, plfd_backbone, handle):

    ext = os.path.splitext(A)
    
    _, typen = ext
    
    if typen == '.jpg' or typen == '.png' or typen == '.jpeg':
        featureA = image_process(args, A, transform, plfd_backbone, handle)
        if featureA is None:
            print("未检测到图片A的人脸")
            return
        featureB = image_process(args, B, transform, plfd_backbone, handle)
        if featureB is None:
            print("未检测到图片B的人脸")
            return 
        score = extract_feature_v2.compare(featureA,featureB)
        if score > 0.6:
            print("是同一个人")
        else:
            print("不是同一个人")
    else:
        print("error")


def main(args):

    checkpoint = torch.load(args.model_path, map_location=device)
    plfd_backbone = PFLDInference().to(device)
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.to(device)
    transform = transforms.Compose([transforms.ToTensor()])
    #videoCapture = cv2.VideoCapture(args.video_path)

    #videoWriter = cv2.VideoWriter("./video/result.avi",cv2.VideoWriter_fourcc('X','V','I','D'),fps,size)
    handle = extract_feature_v2.initHandel(args.extract_model_path)
    
 
    run(args, args.imageA, args.imageB, transform, plfd_backbone, handle)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        default='checkpoint/snapshot/checkpoint.pth.tar',
        type=str)
    parser.add_argument(
        '--extract_model_path',
        default='checkpoint/Backbone_IR_152.pth',
        type=str)
    parser.add_argument(
        '--imageA',
        type=str,
        default='image')
    parser.add_argument(
        '--imageB',
        type=str,
        default='image')
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
 
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)