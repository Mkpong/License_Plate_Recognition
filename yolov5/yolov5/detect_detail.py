import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess

weight_path = './runs/train/yolov5s_test/weights/best.pt'
img_size = 640
confidence = 0.5
source_path = '../test/1/img2.jpg'

subprocess.run(['python', 'detect.py', '--weight', weight_path, '--img', str(img_size), '--conf', str(confidence), '--source', source_path])
subprocess.run(['python', 'detect.py', '--weight', weight_path, '--img', str(img_size), '--conf', str(confidence), '--source', source_path])

imgPath = "./img/number/test.jpg"

# 번호판 이미지 불러오기
img_ori = cv2.imread(imgPath)

height, width, channel = img_ori.shape

# RGB to Gray
gray = cv2.cvtColor(img_ori , cv2.COLOR_BGR2GRAY)

# GaussianBlur
img_blurred = cv2.GaussianBlur(gray, ksize=(5,5) , sigmaX=0)

# Threshold
img_blur_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)
img_thresh = cv2.adaptiveThreshold(
    gray,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)

#Contours
contours, _ = cv2.findContours(
    img_blur_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel) , dtype=np.uint8)
cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255,255,255))

print_image = temp_result
# cv2.imshow("contours" , temp_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Rectangle
temp_result = np.zeros((height, width, channel) , dtype=np.int8)
contours_dict = []
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x,y) , pt2=(x+w, y+h) , color=(255,255,255) , thickness=1)
    
    contours_dict.append({
        'contour':contour,
        'x':x,
        'y':y,
        'w':w,
        'h':h,
        'cx':x+(w/2),
        'cy':y+(h/2)
    })
    
# cv2.imshow("Rectangle1" , temp_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 실제 카메라 해상도에 따른 사이즈 설정 해주어야 함
# 후보 추려내기1
MIN_AREA = 40
MIN_WIDTH, MIN_HEIGHT=1, 4
MIN_RATIO, MAX_RATIO = 0.2, 2.5

possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']
    
    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)

temp_result = np.zeros((height, width, channel), dtype = np.uint8)

for d in possible_contours:
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=1)
    
# cv2.imshow("check1" , temp_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 후보 추려내기 2
MAX_DIAG_MULTIPLYER = 8
MAX_ANGLE_DIFF = 15
MAX_AREA_DIFF = 1
MAX_WIDTH_DIFF = 1.5
MAX_HEIGHT_DIFF = 0.3
MIN_N_MATCHED = 4


possible_contours = sorted(possible_contours, key=lambda x: x['x'])


def find_chars(contour_list):
    matched_result_idx = []
    for i,d1 in enumerate(contour_list):
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                matched_contours_idx.append(d1['idx'])
                continue
                
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])
            
            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
            
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']
            
            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])
                
        # print("len : ", len(matched_contours_idx) , " , " , matched_contours_idx)
        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue
        matched_result_idx.append(matched_contours_idx)
        
        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])
        # print("Unmatched : " , unmatched_contour_idx)
        
        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        
        
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)
        
        break
    return matched_result_idx

result_idx = find_chars(possible_contours)

# 추려진 idx로 contour리스트를 생성한다.
matched_result = []
for idx_list in result_idx:
    box = []
    for j in idx_list:
        for contours in possible_contours:
            if(contours['idx'] == j):
                box.append(contours)
    matched_result.append(box)
    
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    for d in r:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=1)
    # cv2.imshow("check2" , temp_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



# 겹치는 사각형 제거하기
def remove_inner_rectangles(rectangles):
    result = []
    for i in rectangles:
        exresult = []
        for j, mainR in enumerate(i):
            x1, y1, w1, h1 = mainR['x'] , mainR['y'], mainR['w'] ,mainR['h']
            is_inner = False
            for k, subR in enumerate(i):
                if j!=k:
                    x2, y2, w2, h2 = subR['x'], subR['y'], subR['w'], subR['h']
                    if x2 <= x1 and y2 <= y1 and (x2 + w2) >= (x1 + w1) and (y2 + h2) >= (y1 + h1):
                        is_inner = True
                        break
            if not is_inner:
                exresult.append(mainR)
        if(len(exresult) > 5):
            result.append(exresult)
        print("ResultSize : " , len(exresult))
    return result

final_result = remove_inner_rectangles(matched_result)

# 후보군을 확인 1
for r in final_result:
    r = sorted(r, key=lambda d: d['x'])
    img_box = np.zeros((height, width, channel), dtype=np.uint8)
    img_box = img_blur_thresh.copy()
    for d in r:
        cv2.rectangle(img_box, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=1)
    # cv2.imshow("first_result" , img_box)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        

# possible_contours = sorted(possible_contours, key=lambda x: x['x'])
# for i in possible_contours:
#     print("x :",i['x'],"y :",i['y'],"w :",i['w'],"h: ",i['h'],"idx :",i['idx'])

temp_size = 0
for i in final_result:
    if(5 <= len(i) <= 8 and temp_size < len(i)):
        subanswer = i
        temp_size = len(subanswer)
        
img_box = np.zeros((height, width, channel), dtype=np.uint8)
img_box = img_blur_thresh.copy()
if subanswer is not None:
    for d in subanswer:
        cv2.rectangle(img_box, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=1)
    
    
# cv2.imshow("SubAnswer", img_box)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



sumW = 0
sumH = 0
for i in final_result[0]:
    sumW += i['w']
    sumH += i['h']

avgW = sumW / len(final_result[0])
avgH = sumH / len(final_result[0])



for i in range(len(possible_contours)-1):
    x1, y1, w1, h1 = possible_contours[i]['x'], possible_contours[i]['y'], possible_contours[i]['w'], possible_contours[i]['h']
    x2, y2, w2, h2 = possible_contours[i+1]['x'], possible_contours[i+1]['y'], possible_contours[i+1]['w'] , possible_contours[i+1]['h']
    cx1 = x1+w1/2
    cx2 = x2+w2/2
    cy1 = y1+h1/2
    cy2 = y2+h2/2
    if(abs(h1-h2)/h1 < 0.3 and abs(cy1-cy2) < max(h1,h2)/2):
        if(w1+w2 < 1.3*avgW and w1+w2 > 0.7*avgW and abs(cx1-cx2)<1.3*((w1+w2)/2)):
            nx = x1
            ny = min(y1,y2)
            nw = x2+w2-x1
            nh = max(y1+h1,y2+h2)-min(y1,y2)
            ncx = (x1+x2+w2)/2
            ncy = (min(y1,y2)+max(y1+h1,y2+h2))/2
            nidx = len(possible_contours)
            new_contour = {'x' : nx, 'y' : ny , 'w': nw, 'h' : nh , 'cx' : ncx , 'cy' : ncy ,'idx': nidx}
            if(0.5*avgW <= nw <= 1.5*avgW and 0.5*avgH <= nh <= 1.5*avgH):
                possible_contours.append(new_contour)
    if(abs(w1-w2)/w1 < 0.3 and abs(cx1-cx2) < max(w1,w2)/2):
        if(h1+h2 < 1.1*avgH and h1+h2 > 0.7*avgH and abs(cy1-cy2)<(1.3)*((h1+h2)/2)):
            nx = min(x1,x2)
            ny = min(y1,y2)
            nw = max(x1+w1,x2+w2)-nx
            nh = max(y1+h1,y2+h2)-ny
            ncx = (2*nx+nw)/2
            ncy = (2*ny+nh)/2
            nidx = len(possible_contours)
            new_contour = {'x':nx,'y':ny,'w':nw,'h':nh,'cx':ncx,'cy':ncy,'idx':nidx}
            if(0.5*avgW <= nw <= 1.5*avgW and 0.5*avgH <= nh <= 1.5*avgH):
                possible_contours.append(new_contour)
            
            
possible_contours = sorted(possible_contours, key=lambda x: x['x']) 

temp_result = np.zeros((height, width, channel), dtype = np.uint8)

for d in possible_contours:
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=1)
    
# cv2.imshow("check1" , temp_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


    
result_idx = find_chars(possible_contours)


matched_result = []
for idx_list in result_idx:
    box = []
    for j in idx_list:
        for contours in possible_contours:
            if(contours['idx'] == j):
                box.append(contours)
    matched_result.append(box)
    
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    for d in r:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=1)
    # cv2.imshow("check2" , temp_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
final_result = remove_inner_rectangles(matched_result)

# 후보군 출력하기 2
for r in final_result:
    r = sorted(r, key=lambda d: d['x'])
    img_box = np.zeros((height, width, channel), dtype=np.uint8)
    img_box = img_blur_thresh.copy()
    for d in r:
        cv2.rectangle(img_box, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=1)
    # cv2.imshow("second_result" , img_box)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

answer = None
temp_size = 0
for i in final_result:
    if(7 <= len(i) <= 8 and temp_size < len(i)):
        answer = i
        temp_size = len(answer)
        
img_box = np.zeros((height, width, channel), dtype=np.uint8)
img_box = img_blur_thresh.copy()
if answer is not None:
    for d in answer:
        cv2.rectangle(img_box, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=1)

# cv2.imshow("Answer" , img_box)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),
     transforms.Resize((32, 32) , antialias=True)]
)

# Resnet Model Define
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(512, 10)

# Resnet Model Loading
model.load_state_dict(torch.load('./resnet_model20.pth'))

# gpu setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.eval()

number_result = []

if answer is None:
    print("번호판을 인식할 수 없습니다.")
else:
    for d in answer:
        x, y, w, h = d['x'], d['y'], d['w'], d['h']
        image = img_ori[y:y+h, x:x+w]
        # cv2.imshow("checkNumber" , image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        image = transforms.ToPILImage()(image)  # 이미지를 PIL 이미지로 변환
        
        image = transform_test(image)
        image = image.unsqueeze(0)  # 이미지에 배치 차원 추가
        image = image.to(device)  # 디바이스로 이동
        
        with torch.no_grad():
            output = model(image)
        
        predicted_label = torch.argmax(output, dim=1)
        # print("Predicted Label:", predicted_label.item())
        number_result.append(predicted_label.item())

    print("RESULT : " , number_result)