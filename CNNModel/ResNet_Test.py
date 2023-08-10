import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Resnet Model Define
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(512, 2)

# gpu setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 데이터 전처리
transform = transforms.Compose(
    [transforms.ToTensor(),                         #텐서로 변환 
     transforms.Normalize((0.5,), (0.5,)),          #정규화
     transforms.Resize((128 ,128) , antialias=True)]  #28x28 resize
)

model.load_state_dict(torch.load('./models/resnet/resnet_plate.pth'))
image = cv2.imread("img0.jpg")
# 모델 평가
model.eval()


    
image = transforms.ToPILImage()(image)  # 이미지를 PIL 이미지로 변환
image = transform(image)
plt.imshow(image.permute(1, 2, 0))  # 채널 순서 변경 (C, H, W) -> (H, W, C)
plt.show()
image = image.unsqueeze(0)  # 이미지에 배치 차원 추가
image = image.to(device)  # 디바이스로 이동

output = model(image)

predicted_label = torch.argmax(output, dim=1)
print("Predicted Label:", predicted_label.item())