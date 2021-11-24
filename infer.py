import argparse
import os
import sys
import cv2
import json
from pathlib import  Path
import torch
import torch.backends.cudnn as cudnn
from models.yolo import Detect, Model
from utils.datasets import LoadImages
from utils.torch_utils import time_sync
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

from models.common import DetectMultiBackend

# param
conf_thres = 0.25
iou_thres = 0.45
classes = None
agnostic_nms = False
max_det = 999
img_size = [640, 640]
device = 'cuda:0'

#path
image_dir = 'dataset/test/'
model_path = 'best.pt'
save_dir = 'result/'
filepath = 'result/labels/'
#answer
data = []
result_to_json = []



model = DetectMultiBackend(model_path, device=device, dnn=False)
model.model.float()
model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  # warmup

#load image
dataset = LoadImages(image_dir, img_size=img_size, stride=32)

for path, img, im0s, vid_cap, s in dataset:
    img = torch.from_numpy(img).to(device)
    img =img.float()
    img /= 255
    img = img[None]
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    for i, det in enumerate(pred):
        p, im0s = path, im0s.copy()

    p = Path(p)  # to Path
    txt_path = save_dir + 'labels/' + p.stem
    #txt_path = str(save_dir / 'labels' / p.stem) + ('')  # im.txt
    gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
# write to txt file
    if not os.path.isdir('result/labels'):
        os.makedirs('result/labels')
    for *xyxy, conf, cls in reversed(det):
                    if True:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
# dump to answer.json
all_file = os.listdir(filepath)
for file in all_file:
    f = open(filepath+file,'r')
    img_name = str(file[:-4])+'.png'
    img_id = int(file[:-4])
    im = cv2.imread(image_dir+img_name)
    h, w, cc = im.shape
    contents = f.readlines()
    for content in contents:
        tmp = {}
        content = content.replace('\n','')
        c = content.split(' ')
        tmp["image_id"] = img_id
        tmp['category_id'] = (int(c[0]))
        w_center = w*float(c[1])
        h_center = h*float(c[2])
        width = w*float(c[3])
        height = h*float(c[4])
        left = float(w_center - width/2)
        top = float(h_center - height/2)
        tmp['bbox'] = (tuple((left, top, width, height)))
        tmp['score'] = (float(c[5]))
        print(tmp)
        result_to_json.append(tmp)
f.close()
ret = json.dumps(result_to_json, indent=4)

print(len(result_to_json))
with open('answer.json', 'w') as fp:
    fp.write(ret)   