import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Resnet Model Define
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(512, 10)

model2 = models.resnet18(weights=None)
model2.conv1 = nn.Conv2d(3,64, kernel_size=7, stride=2, padding=3, bias=False)
model2.fc=nn.Linear(512,32)

# Resnet Model Loading
model.load_state_dict(torch.load('./models/resnet/resnet_number.pth'))
model2.load_state_dict(torch.load('./models/resnet/resnet_korean.pth'))

# gpu setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model2.to(device)

# criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.eval()
model2.eval()

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),
     transforms.Resize((32, 32) , antialias=True)]
)

def test(image):
    image = transforms.ToPILImage()(image)
    image = transform_test(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model(image)
    predicted_label = torch.argmax(output, dim=1)
    print(predicted_label.item())
    
def test2(image):
    image = transforms.ToPILImage()(image)
    image = transform_test(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model2(image)
    predicted_label = torch.argmax(output, dim=1)
    print(predicted_label.item())
    
# net = cv2.dnn.readNet("yolov4-tiny-custom_final.weights" , "yolov4-tiny-custom.cfg")
# classes = []
# with open("ClassNames.names") as f:
#     classes = [line.strip() for line in f.readlines()]
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0,255,size=(len(classes) , 3))

# img_ori = cv2.imread("./wide/img21.jpg")
# height, width, channels = img_ori.shape

# blob = cv2.dnn.blobFromImage(img_ori, 0.00392 , (416, 416) , (0,0,0) , True , crop=False)
# net.setInput(blob)
# outs = net.forward(output_layers)

# class_ids = []
# confidences = []
# boxes = []
# for out in outs:
#     for detection in out:
#         scores = detection[5:]
#         class_id = np.argmax(scores)
#         confidence = scores[class_id]
#         if confidence > 0.5:
#             # Object detected
#             center_x = int(detection[0] * width)
#             center_y = int(detection[1] * height)
#             w = int(detection[2] * width)
#             h = int(detection[3] * height)
#             # Rectangle coordinates
#             x = int(center_x - w / 2)
#             y = int(center_y - h / 2)
#             boxes.append([x, y, w, h])
#             confidences.append(float(confidence))
#             class_ids.append(class_id)
            
# indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# font = cv2.FONT_HERSHEY_PLAIN
# img_detect_1 = img_ori.copy()
# for i in range(len(boxes)):
#     if i in indexes:
#         x, y, w, h = boxes[i]
#         # label = str(classes[class_ids[i]])
#         color = colors[class_ids[i]]
#         cv2.rectangle(img_detect_1, (x, y), (x + w, y + h), color, 2)
#         # cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
        
#         # 박스 영역 추출
#         license_plate_img = img_ori[y-10:y+h+10, x-10:x+w+10]
        
# cv2.imshow("license_plate_img", license_plate_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

net = cv2.dnn.readNet("./NumberWordDetect/yolov4-tiny-custom_final.weights" , "./NumberWordDetect/yolov4-tiny-custom.cfg")
classes = []
with open("./NumberWordDetect/ClassNames.names") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes) , 3))

license_plate_img = cv2.imread("img1.jpg")
height, width, channels = license_plate_img.shape

blob = cv2.dnn.blobFromImage(license_plate_img, 0.00392 , (320, 320) , (0,0,0) , True , crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

print(len(indexes))
img_detect_2 = license_plate_img.copy()
boxes = sorted(boxes, key=lambda x: x[0])
cnt = 0
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        if(w <= 0 or h <= 0 or x<0 or y<0):
            continue
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img_detect_2, (x, y), (x + w, y + h), color, 1)
        # cv2.putText(img_detect_2, label, (x, y + 30), font, 3, color, 3)
        
        # 박스 영역 추출
        result = img_detect_2[y:y+h, x:x+w]
        # cv2.imshow("Image", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("ASD", end="")
        if(class_ids[i] == 1):
            test(result)
        else:
            test2(result)


cv2.imshow("img_detect_2", img_detect_2)
cv2.waitKey(0)
cv2.destroyAllWindows()