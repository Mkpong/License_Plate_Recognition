# 가우시안블러 처리 후 학습


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image
import cv2

# Resnet Model Define
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(512, 10)

# gpu setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 데이터 전처리
transform = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.ToTensor(),                         #텐서로 변환 
     transforms.Normalize((0.5,), (0.5,)),          #정규화
     transforms.Resize((32 ,32) , antialias=True)]  #28x28 resize
)

# file open
with open("./data/test/image.txt") as f:
    imagesPath = f.read().splitlines()
with open('./data/test/label.txt') as f:
    labelsPath = f.read().splitlines()

# 전처리된 이미지 저장 리스트
PreprocessingImage = []

# 전처리 수행
for img_path in imagesPath:
    image = cv2.imread(img_path)  # 이미지 파일 열기
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, ksize=(5,5) , sigmaX=0)
    image = cv2.adaptiveThreshold(
    image,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
    )
    print_image = image
    image = transform(image)
    PreprocessingImage.append(image)

# 텐서 구조 출력
cv2.imshow("TEST" , print_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

PreprocessingImage = torch.stack(PreprocessingImage)
labels = [int(label) for label in labelsPath]
labels = torch.tensor(labels)

# dataset & dataloader(Train)
train_dataset = list(zip(PreprocessingImage , labels))
train_dataloader = torch.utils.data.DataLoader(train_dataset , batch_size=4, shuffle=True)

# dataset & dataloader(test)

# Training
num_epochs = 16
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataloader.dataset)
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    
# 모델 저장
torch.save(model.state_dict(), "./models/resnet/resnet_model20.pth")

# 모델 평가
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")