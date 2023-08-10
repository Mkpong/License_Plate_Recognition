import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image
import cv2

# Resnet Model Define
model = models.resnet34(pretrained = True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 6)
print(model.fc.in_features)
# gpu setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 데이터 전처리
transform = transforms.Compose(
    [transforms.Resize((32 ,32) , antialias=True),
    transforms.ToTensor(),                         #텐서로 변환 
    transforms.Normalize((0.5,), (0.5,))]          #정규화  #28x28 resize
)

# file open
with open("./data/train_region/test/image.txt") as f:
    imagesPath_test = f.read().splitlines()
with open('./data/train_region/test/label.txt') as f:
    labelsPath_test = f.read().splitlines()
with open('./data/train_region/train/image.txt') as f:
    imagesPath_train = f.read().splitlines()
with open('./data/train_region/train/label.txt') as f:
    labelsPath_train = f.read().splitlines()

# 전처리된 이미지 저장 리스트
PreprocessingImage_train = []
PreprocessingImage_test = []

# 전처리 수행
for img_path in imagesPath_train:
    image = Image.open(img_path)  # 이미지 파일 열기
    image = image.convert("RGB")
    image = transform(image) # 전처리 수행
    PreprocessingImage_train.append(image)
    
for img_path in imagesPath_test:
    image = Image.open(img_path)  # 이미지 파일 열기
    image = image.convert("RGB")
    image = transform(image) # 전처리 수행
    PreprocessingImage_test.append(image)

PreprocessingImage_train = torch.stack(PreprocessingImage_train)
PreprocessingImage_test = torch.stack(PreprocessingImage_test)
labels_train = [int(label) for label in labelsPath_train]
labels_train = torch.tensor(labels_train)

labels_test = [int(label) for label in labelsPath_test]
labels_test = torch.tensor(labels_test)

# dataset & dataloader(Train)
train_dataset = list(zip(PreprocessingImage_train , labels_train))
train_dataloader = torch.utils.data.DataLoader(train_dataset , batch_size=16, shuffle=True)

test_dataset = list(zip(PreprocessingImage_test , labels_test))
test_dataloader = torch.utils.data.DataLoader(test_dataset , batch_size = 16, shuffle=True)


# dataset & dataloader(test)

# Training
num_epochs = 50
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
torch.save(model.state_dict(), "./models/resnet/resnet34_region.pth")

# 모델 평가
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")