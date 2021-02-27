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

from concurrent.futures import ThreadPoolExecutor, wait

from pfld.pfld import PFLDInference, AuxiliaryNet

from extract_feature import model_irse
from extract_feature import landmarks_alignment
from extract_feature import extract_feature_v2
import dlib
import threading
from tqdm import tqdm
import skvideo.io
import imutils
import json
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_num(point_dict,name,axis):
    num = point_dict.get(f'{name}')[axis]
    num = float(num)
    return num

def cross_point(line1, line2):  
    x1 = line1[0]  
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1) 
    b1 = y1 * 1.0 - x1 * k1 * 1.0  
    if (x4 - x3) == 0: 
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]

def point_line(point,line):
    x1 = line[0]  
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    x3 = point[0]
    y3 = point[1]

    k1 = (y2 - y1)*1.0 /(x2 -x1) 
    b1 = y1 *1.0 - x1 *k1 *1.0
    k2 = -1.0/k1
    b2 = y3 *1.0 -x3 * k2 *1.0
    x = (b2 - b1) * 1.0 /(k1 - k2)
    y = k1 * x *1.0 +b1 *1.0
    return [x,y]

def point_point(point_1,point_2):
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
    return distance
def get_param(args, line, feacRect, simg, transform, plfd_backbone, handle, edge = 20):
   #cv2.rectangle(img,  tuple([feacRect.rect.left(), feacRect.rect.top()]), tuple([feacRect.rect.right(), feacRect.rect.bottom()]), (0, 255, 255), 2)
    height, width = simg.shape[:2]
    top = feacRect.top()-edge
    if top < 0:
        top = 0
    
    left = feacRect.left()-edge
    if left < 0:
        left = 0
        
    right = feacRect.right()+edge
    if right > width:
        right = width
        
    bottom = feacRect.bottom()+edge
    if bottom > height:
        bottom = height
   
   
    img = simg[top:bottom,left:right]
    #img =cv2.imread('1.jpg')

    model_test = AntiSpoofPredict(args.device_id)
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
    #cv2.imwrite('1.jpg', cropped1)
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
    point_dict = {}
    i = 0
    
    #face align


    src_list = pre_landmark.tolist()
    RotateMatrix = landmarks_alignment.warp_im(src_list)
    rev_aligned = landmarks_alignment.alignlandmarks(cropped, RotateMatrix)
    #cv2.imwrite('test.jpg', rev_aligned)
    
    
    for (x,y) in pre_landmark.astype(np.float32):
        point_dict[f'{i}'] = [x,y]
        i += 1
                
    feature = extract_feature_v2.extract_feature(rev_aligned, handle)
    #print(feature)
    feature = feature.tolist()
    
    #strf = ''.join(str(feature).lstrip('[').rstrip(']'))
    #print(strf)
    #print(line)
    line['Feature'] = feature
    #filename.write(iname)
    #print(point_dict)

    #filename.write('\n')        
    if args.pose:
        #yaw
        point1 = [get_num(point_dict, 1, 0), get_num(point_dict, 1, 1)]
        point31 = [get_num(point_dict, 31, 0), get_num(point_dict, 31, 1)]
        point51 = [get_num(point_dict, 51, 0), get_num(point_dict, 51, 1)]
        crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
        yaw_mean = point_point(point1, point31) / 2
        yaw_right = point_point(point1, crossover51)
        yaw = (yaw_mean - yaw_right) / yaw_mean
        yaw = int(yaw * 71.58 + 0.7037)

        #pitch
        pitch_dis = point_point(point51, crossover51)
        if point51[1] < crossover51[1]:
            pitch_dis = -pitch_dis
        pitch = int(1.497 * pitch_dis + 18.97)

        #roll
        roll_tan = abs(get_num(point_dict,60,1) - get_num(point_dict,72,1)) / abs(get_num(point_dict,60,0) - get_num(point_dict,72,0))
        roll = math.atan(roll_tan)
        roll = math.degrees(roll)
        if get_num(point_dict, 60, 1) > get_num(point_dict, 72, 1):
            roll = -roll
        roll = int(roll)
        #cv2.putText(img,f"Head_Yaw(degree): {yaw}",(30,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
        #cv2.putText(img,f"Head_Pitch(degree): {pitch}",(30,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
        #cv2.putText(img,f"Head_Roll(degree): {roll}",(30,150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
        
        line["Rotate"] = [pitch,yaw,roll]
    #filename.write('\n')
    if args.landmarks:
        marks = []
        for i in range(len(point_dict)):
            #print(point_dict[f'{i}'])
            point_dict[f'{i}'][0] = point_dict[f'{i}'][0] + feacRect.left()-edge
            point_dict[f'{i}'][1] = point_dict[f'{i}'][1] + feacRect.top()-edge
            if i < len(point_dict):
                #iname = "{},".format(str(point_dict[f'{i}']).lstrip('[').rstrip(']'))
                marks.append(point_dict[f'{i}'])
            else:
                #iname = "{}:".format(str(point_dict[f'{i}']).lstrip('[').rstrip(']'))
                marks.append(point_dict[f'{i}'])
        line["Landmarks"] = marks 
        #filename.write('\n')
    return line


def test_rotate(video):
    metadata = skvideo.io.ffprobe(video)
    rotate = 0
    #print(metadata['video'])
    try:
        d = metadata['video'].get('tag')[0]
        if d.setdefault('@key') == 'rotate': #获取视频自选择角度
            rotate = 360-int(d.setdefault('@value'))
    except:
        return None,None
    return d,rotate
        
def video_process(args, inputname, video, transform, plfd_backbone, handle, face_detector):
    videoCapture = cv2.VideoCapture(inputname)
    #cv2.imwrite("1.jpg",img)
    if not videoCapture.isOpened():
        return
    d, rotate = test_rotate(inputname)

    #print(rotate)
    
    
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_num = int(videoCapture.get(7))
    save_num = frame_num

    if frame_num <= args.stips:
        print("跳帧数超过总视频帧数，请减小strips参数值")
        return

    print("size:",frame_num)

    data = {
        "AllFrameNum":0,
        "Pose":False,
        "Axis":False,

    }

    #data = json.dumps(odci)

    if args.suffix is not None:
        nametxt = 'json/{}_{}.json'.format(video[:-4], args.suffix)
    else:
        nametxt = 'json/{}.json'.format(video[:-4])
    print(nametxt)

    pose = ''
    land = ''
    if args.landmarks:
        data["Axis"] = True
    
    if args.pose:
        data["Pose"] = True



    if args.stips:
        frame_num = math.ceil(frame_num/(args.stips+1))

    data['AllFrameNum'] = frame_num

    save_id = 0 

    for idx in tqdm(range(save_num),ascii=True,desc=nametxt):
    
        success,simg = videoCapture.read()
        if d is not None and rotate:
            simg = imutils.rotate(simg, 360-int(d.setdefault('@value')))
        if success:
            if idx%(args.stips+1) == 0:

                save_id = save_id + 1
                face_rects = face_detector(simg, 0)
                #framenum = data['FrameNum'][f'{save_id}']


                #filename.write(iname)
                #filename.write('\n') 

                for line, feacRect in enumerate(face_rects):
                    #framenum[f'{line}'] = line
                    data.setdefault(f'FrameNum_{save_id}', {})[f'FaceNum_{line}'] = {}
                    get_param(args, data.setdefault(f'FrameNum_{save_id}', {})[f'FaceNum_{line}'], feacRect, simg,transform, plfd_backbone, handle)
                    #t1 = threading.Thread(target=get_param, args=(args, line, feacRect, simg,transform, plfd_backbone, handle, filename))
                    #t1.start()

            else:
                continue
                
        else:
            break


    json_str = json.dumps(data)
    with open(nametxt, 'w') as json_file:
        json_file.write(json_str)

def run(args, inputname, video, transform, plfd_backbone, handle, face_detector):



    ext = os.path.splitext(video)
    
    _, typen = ext
    
    if typen == '.avi' or typen == '.mp4' or typen == '.MP4':
        video_process(args, inputname, video, transform, plfd_backbone, handle, face_detector)

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
    
    # Create dlib face detector (should be replaced by RetinaFace)
    face_detector = dlib.get_frontal_face_detector()
    pool = ThreadPoolExecutor(max_workers=3)

    futures = []
    for video in os.listdir(args.video_path):
        name = os.path.join(args.video_path, video)
        print(name)
        run(args, name, video, transform, plfd_backbone, handle, face_detector)

    
    
 

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
        '--video_path',
        type=str,
        default='video')
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--pose",
        action='store_true',
        help="pose xroll yroll  zroll")
    parser.add_argument(
        "--landmarks",
        action='store_true',
        help="lanmarks")
    parser.add_argument(
        "--stips",
        default=100,
        help="frame tips",
        type=int)
    parser.add_argument(
        "--suffix",
        default=None,
        help="suffix name",
        type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
