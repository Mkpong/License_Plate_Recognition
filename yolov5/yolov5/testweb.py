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
import numpy
import json



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

weights_plate = "C:/Users/leeja/Github_local_repository/License_Plate_Recognition/yolov5/yolov5/runs/train/plate_detect_korea400/weights/best.pt"
device = select_device("")
model_plate = DetectMultiBackend(weights_plate, device=device, dnn=False, data='data/coco128.yaml', fp16=False)

weights_numberword = "C:/Users/leeja/Github_local_repository/License_Plate_Recognition/yolov5/yolov5/runs/train/numberWord_detect900/weights/best.pt"
device = select_device("")
model_numberword = DetectMultiBackend(weights_numberword, device=device, dnn=False, data='data/coco128.yaml', fp16=False)

@smart_inference_mode()
def run_plate(
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

    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    stride_plate, names_plate, pt_plate = model_plate.stride, model_plate.names, model_plate.pt
    imgsz = check_img_size(imgsz, s=stride_plate)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride_plate, auto=pt_plate, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride_plate, auto=pt_plate)
    else:
        if crop_image is None:
            dataset = LoadImages(source, img_size=imgsz, stride=stride_plate, auto=pt_plate, vid_stride=vid_stride)
        else:
            resize_image = cv2.resize(crop_image, (640, 640))
            resize_image = np.transpose(resize_image , (2,0,1))
            dataset = [("a",resize_image, crop_image, None, None)]
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model_plate.warmup(imgsz=(1 if pt_plate or model_plate.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path,im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model_plate.device)
            im = im.half() if model_plate.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model_plate(im, augment=augment, visualize=visualize)

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
            annotator = Annotator(im0, line_width=line_thickness, example=str(names_plate))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    crop_img = im0[y1:y2, x1:x2]
                    result_list.append([x1,y1,x2,y2,conf.item(),int(cls)])
                    cv2.imwrite("./box/output.jpg" , crop_img)

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None # if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
            # Stream results
            im0 = annotator.result()

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    return result_list


@smart_inference_mode()
def run_numberword(
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

    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    stride_numberword, names_numberword, pt_numberword = model_numberword.stride, model_numberword.names, model_numberword.pt
    imgsz = check_img_size(imgsz, s=stride_numberword)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride_numberword, auto=pt_numberword, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride_numberword, auto=pt_numberword)
    else:
        if crop_image is None:
            dataset = LoadImages(source, img_size=imgsz, stride=stride_numberword, auto=pt_numberword, vid_stride=vid_stride)
        else:
            resize_image = cv2.resize(crop_image, (640, 640))
            resize_image = np.transpose(resize_image , (2,0,1))
            dataset = [("a",resize_image, crop_image, None, None)]
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model_numberword.warmup(imgsz=(1 if pt_numberword or model_numberword.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path,im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model_numberword.device)
            im = im.half() if model_numberword.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model_numberword(im, augment=augment, visualize=visualize)

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
            annotator = Annotator(im0, line_width=line_thickness, example=str(names_numberword))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    crop_img = im0[y1:y2, x1:x2]
                    result_list.append([x1,y1,x2,y2,conf.item(),int(cls)])
                    cv2.imwrite("./box/output.jpg" , crop_img)

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None # if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
            # Stream results
            im0 = annotator.result()
        # cv2.imshow("Result" , im0)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
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

model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(512, 10)

model2 = models.resnet18(weights=None)
model2.conv1 = nn.Conv2d(3,64, kernel_size=7, stride=2, padding=3, bias=False)
model2.fc=nn.Linear(512,40)

model3 = models.resnet18(weights=None)
model3.conv1 = nn.Conv2d(3,64, kernel_size=7, stride=2, padding=3, bias=False)
model3.fc=nn.Linear(512,4)

# Resnet Model Loading
model.load_state_dict(torch.load('./resnet/resnet18_number.pth'))
model2.load_state_dict(torch.load('./resnet/resnet18_word.pth'))
model3.load_state_dict(torch.load('./resnet/resnet18_region.pth'))

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

app = Flask(__name__)                                          
plateTime = 0
numberWordTime = 0
resnetTime = 0
cnt = 0
@app.route("/test", methods=['post'])
def number_detect():
    getimage_start = time.time()
    if 'image' not in request.files:
        print("Not image")
        return "Fail"
    image_file = request.files['image']
    image = cv2.imdecode(numpy.frombuffer(image_file.read(), numpy.uint8), cv2.IMREAD_COLOR)
    getimage_end = time.time()
    print("image loading time : ", getimage_end-getimage_start)
    cv2.imwrite("./box/output.jpg" , image)
    imgpath = "./box/output.jpg"
    plate_start = time.time()
    plate_result = run_plate(source=imgpath)
    
    if len(plate_result) == 0:
        print("번호판을 찾을 수 없습니다.")
        data = {
        "time": "Fail",
        "carnumber": "",
        "state": ""
        }
        json_data = json.dumps(data)
        return json_data
    else:
        plate_result.sort(key=lambda x: x[4] , reverse=True)
        plate = plate_result[0]
        plate_type = plate[5]

    type = "null"
    if plate_type == 0:
        type = "일반"
    elif plate_type == 1:
        type = "친환경"
    elif plate_type == 2:
        type = "유럽식 운수용"
    elif plate_type == 3:
        type = "미국식 운수용"
    else:
        print("기타")
    plate_end = time.time()
    print("plateTime : " , plate_end-plate_start)
    
    numberword_start = time.time()
    imgpath ="./box/output.jpg"
    img_ori = cv2.imread(imgpath)
    result = run_numberword(source=imgpath)
    numberword_end = time.time()
    print("numberWord : " , numberword_end-numberword_start)
    
    transform_test = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((32, 32) , antialias=True)]
    )


    number_result = []
    detection_result = []
    
    # 넓이의 80% 이상이 중복되면 박스 하나를 제거하는 코드
    for i in range(len(result)):
        main = result[i]
        for j in range(i, len(result) , 1):
            main = result[i]
            
    
    resnet_start = time.time()
    if result is None:
        print("번호판을 인식할 수 없습니다.")
        data = {
        "time": "NO_DETECTION",
        "carnumber": "",
        "state": ""
        }
        json_data = json.dumps(data)
        return json_data
    elif (plate_type == 0 or plate_type == 1) and len(result) > 6 and len(result) < 9:
        result = sorted(result, key=lambda x: x[0])
        # print(result)
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
        print(result)
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
        print("번호판을 인식할 수 없습니다.(인식 오류)")
        data = {
        "time": "NO_DETECTION",
        "carnumber": "",
        "state": ""
        }
        json_data = json.dumps(data)
        return json_data
        
    resnet_end = time.time()
    print("resnet : " , resnet_end-resnet_start)
    text = ''.join(detection_result)
    current_time = datetime.datetime.now()
    nowTime = current_time.strftime("%Y-%m-%d %H:%M:%S")
    url = "http://localhost:8080/api/detect"
    json_Data = {"result":text, "time":nowTime, "type":plate_type}
    response = requests.post(url, json={"result":text, "time":nowTime , "type":plate_type})
    response_data = response.json()
    print(response_data)
    global cnt
    global plateTime
    global numberWordTime
    global resnetTime
    cnt += 1
    plateTime += plate_end-plate_start
    numberWordTime += numberword_end-numberword_start
    resnetTime += resnet_end-resnet_start
    if(response_data['state'] == "entrance"):
        data = {
        "time": nowTime,
        "carnumber": text,
        "state": response_data['ticket'],
        "type": type
        }
        print(text)
        json_data = json.dumps(data , ensure_ascii=False)
    else:
        data = {
            "time": response_data["parkingTime"],
            "carnumber": text,
            "state": response_data['ticket'] if response_data['ticket']=="정기권" else str(response_data['parkingFee']),
            "type": type
        }
        print(text)
        json_data = json.dumps(data , ensure_ascii=False)
    print(json_data)
    return json_data

@app.route("/result")
def result():
    return "plateTime : "+str(plateTime/cnt)+"\n"+"numberWordTime : "+str(numberWordTime/cnt)+"\n"+"resnetTime : "+str(resnetTime/cnt)

if(__name__) == '__main__':
    app.run(host='0.0.0.0')