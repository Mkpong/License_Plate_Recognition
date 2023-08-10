import os
import platform
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request
import datetime
import requests
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

test_world = "test"

model = models.resnet34(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(512, 10)

model2 = models.resnet34(weights=None)
model2.conv1 = nn.Conv2d(3,64, kernel_size=7, stride=2, padding=3, bias=False)
model2.fc=nn.Linear(512,40)

model3 = models.resnet34(weights=None)
model3.conv1 = nn.Conv2d(3,64, kernel_size=7, stride=2, padding=3, bias=False)
model3.fc=nn.Linear(512,4)

# Resnet Model Loading
model.load_state_dict(torch.load('../resnet/resnet_number600.pth'))
model2.load_state_dict(torch.load('../resnet/resnet_korean.pth'))
model3.load_state_dict(torch.load('../resnet/resnet_region.pth'))

# gpu setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model2.to(device)
model3.to(device)

# criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.eval()
model2.eval()
model3.eval()

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        crop_image = None
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt

    
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        if crop_image is None:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        else:
            resize_image = cv2.resize(crop_image, (640, 640))
            resize_image = np.transpose(resize_image , (2,0,1))
            dataset = [("a",resize_image, crop_image, None, None)]
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path,im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        result_list = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # print(f"Box coordinates: {int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}" , "conf:",conf.item())
                    x1, y1, x2, y2 = map(int, xyxy)
                    crop_img = im0[y1:y2, x1:x2]
                    result_list.append([x1,y1,x2,y2,conf.item(),int(cls)])
                    cv2.imwrite("../box/output.jpg" , crop_img)

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None # if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    # if save_crop:
                    #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            # Stream results
            im0 = annotator.result()
        # cv2.imshow("Result" , im0)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    return result_list

def getWord(d):
    if(d == 0):
        return "가"
    elif(d == 1):
        return "나"
    elif(d == 2):
        return "다"
    elif d == 3:
        return "라"
    elif d == 4:
        return "마"
    elif d==5:
        return "거"
    elif d== 6:
        return "너"
    elif d == 7:
        return "더"
    elif d == 8:
        return "러"
    elif d == 9:
        return "머"
    elif d == 10:
        return "버"
    elif d == 11:
        return "서"
    elif d == 12:
        return "어"
    elif d == 13:
        return "저"
    elif d == 14:
        return "고"
    elif d == 15:
        return "노"
    elif d == 16:
        return "도"
    elif d == 17:
        return "로"
    elif d == 18:
        return "모"
    elif d == 19:
        return "보"
    elif d == 20:
        return "소"
    elif d == 21:
        return "오"
    elif d == 22:
        return "조"
    elif d == 23:
        return "구"
    elif d == 24:
        return "누"
    elif d == 25:
        return "두"
    elif d == 26:
        return "루"
    elif d == 27:
        return "무"
    elif d == 28:
        return "부"
    elif d == 29:
        return "수"
    elif d == 30:
        return "우"
    elif d == 31:
        return "주"
    elif d == 32:
        return "바"
    elif d == 33:
        return "사"
    elif d == 34:
        return "아"
    elif d == 35:
        return "자"
    elif d == 36:
        return "하"
    elif d == 37:
        return "허"
    elif d == 38:
        return "호"
    elif d == 39:
        return "배"
    else:
        return "X"
    
def getRegion(d):
    if d == 0 or d == 1:
        return "경기"
    elif d == 2 or d == 3:
        return "서울"

app = Flask(__name__)                                          

@app.route("/detect/<int:N>")
def number_detect(N):
    start = time.time()
    imgpath = "../test/wide/img{}.jpg".format(N)
    img_ori = cv2.imread(imgpath)
    plate_result = run(weights="C:/Users/leeja/Github_local_repository/License_Plate_Recognition/yolov5/yolo/runs/train/plate_detect_korea400/weights/best.pt", source=imgpath)

    print(plate_result)
    if plate_result is None:
        print("번호판을 찾을 수 없습니다.")     
        return "FAIL"
    else:
        plate_result.sort(key=lambda x: x[4] , reverse=True)
        plate = plate_result[0]
        plate_type = plate[5]

    if plate_type == 0:
        print("일반")
    elif plate_type == 1:
        print("전기")
    elif plate_type == 2:
        print("유럽식 운수용")
    elif plate_type == 3:
        print("미국식 운수용")
    else:
        print("기타")
        
    
    imgpath ="../box/output.jpg"
    # imgpath = "./test/test/img100.jpg"
    img_ori = cv2.imread(imgpath)
    # cv2.imshow("image", img_ori)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    result = run(weights="C:/Users/leeja/Github_local_repository/License_Plate_Recognition/yolov5/yolo/runs/train/numberWord_detect900/weights/best.pt", source=imgpath)
    transform_test = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((32, 32) , antialias=True)]
    )
    # Resnet Model Define
    end = time.time()
    number_result = []
    detection_result = []

    if result is None:
        print("번호판을 인식할 수 없습니다.")
    elif (plate_type == 0 or plate_type == 1) and len(result) > 6:
        result = sorted(result, key=lambda x: x[0])
        result.reverse()
        for idx,i in enumerate(result):
            image = img_ori[i[1]:i[3], i[0]:i[2]]
            image = transforms.ToPILImage()(image)  # 이미지를 PIL 이미지로 변환
            image = transform_test(image)
            image = image.unsqueeze(0)  # 이미지에 배치 차원 추가
            image = image.to(device)  # 디바이스로 이동
            
            if(idx != 4):
                with torch.no_grad():
                    output = model(image)
            else:
                with torch.no_grad():
                    output = model2(image)
            
            predicted_label = torch.argmax(output, dim=1)
            # print("Predicted Label:", predicted_label.item())
            number_result.append(predicted_label.item())
        for i,d in enumerate(number_result):
            if(i==4):
                detection_result.append(getWord(d))
            else:
                detection_result.append(str(d))
        detection_result.reverse()
    elif plate_type == 2 and len(result) > 7:
        result = sorted(result, key=lambda x: x[0])
        for idx,i in enumerate(result):
            image = img_ori[i[1]:i[3], i[0]:i[2]]
            image = transforms.ToPILImage()(image)  # 이미지를 PIL 이미지로 변환
            image = transform_test(image)
            image = image.unsqueeze(0)  # 이미지에 배치 차원 추가
            image = image.to(device)  # 디바이스로 이동
            
            if(idx == 0):
                with torch.no_grad():
                    output = model3(image)
            elif idx == 3:
                with torch.no_grad():
                    output = model2(image)
            else:
                with torch.no_grad():
                    output = model(image)
            
            predicted_label = torch.argmax(output , dim=1)
            number_result.append(predicted_label.item())
        for idx,d in enumerate(number_result):
            if idx == 0:
                detection_result.append(getRegion(d))
            elif idx == 3:
                detection_result.append(getWord(d))
            else:
                detection_result.append(str(d))
    elif plate_type == 3 and len(result) > 7:
        result = sorted(result , key=lambda y: y[1])
        top = result[:3]
        bottom = result[3:]
        top = sorted(top , key=lambda x:x[0])
        bottom = sorted(bottom , key=lambda x:x[0])
        box = top.copy()
        box.extend(bottom)
        for idx,i in enumerate(box):
            image = img_ori[i[1]:i[3], i[0]:i[2]]
            image = transforms.ToPILImage()(image)  # 이미지를 PIL 이미지로 변환
            image = transform_test(image)
            image = image.unsqueeze(0)  # 이미지에 배치 차원 추가
            image = image.to(device)  # 디바이스로 이동
            if(idx == 0):
                with torch.no_grad():
                    output = model3(image)
            elif idx == 3:
                with torch.no_grad():
                    output = model2(image)
            else:
                with torch.no_grad():
                    output = model(image)
            predicted_label = torch.argmax(output , dim=1)
            number_result.append(predicted_label.item())
        for idx,d in enumerate(number_result):
            if idx == 0:
                detection_result.append(getRegion(d))
            elif idx == 3:
                detection_result.append(getWord(d))
            else:
                detection_result.append(str(d))
    else:
        print("no_detection")
        result = "no_detection"
            
        
    text = ''.join(detection_result)
    current_time = datetime.datetime.now()
    nowTime = current_time.strftime("%Y-%m-%d %H:%M:%S")
    # url = "http://localhost:8080/api/detect"
    # response = requests.post(url, json={"result":text, "time":nowTime , "type":plate_type})
    if(result == "no_detection"):
        return result


    return text

if(__name__) == '__main__':
    app.run(host='0.0.0.0')