import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image

# CNN Model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.keep_prob = 0.5
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,10,kernel_size = 3, stride=1 , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.fc1 = nn.Linear(576, 625, bias=True)
        self.interfc = torch.nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Linear(625, 256, bias=True)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0) , -1)
        out = self.fc1(out)
        out = self.interfc(out)
        x = self.fc2(out)
        return x

# Image Proprocessing define
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)
transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),
     transforms.Grayscale(),
     transforms.Resize((28, 28) , antialias=True)]
)

# file open
with open("./data/image.txt") as f:
    imagesPath = f.read().splitlines()
with open('./data/label.txt') as f:
    labels_s = f.read().splitlines()

# 전처리된 이미지 저장 리스트
after_image = []

# 전처리 수행
for img_path in imagesPath:
    image = Image.open(img_path)  # 이미지 파일 열기
    image = transform_test(image)  # 전처리 수행
    after_image.append(image)
    

after_image = torch.stack(after_image)
labels_s = [int(label) for label in labels_s]
labels_s = torch.tensor(labels_s)

# dataset & dataloader
test_dataset = list(zip(after_image, labels_s))
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
# test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

# 데이터 로더 생성
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# GPU Activation
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

# net activate and test
net = Net().to(device)
# print(net)

net = Net()
net.load_state_dict(torch.load('./models/model100.pth'))
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# # model Training
# for epoch in range(100):
#     running_loss = 0.0
#     for images, labels in train_dataloader:

#         images = images.to(device)
#         labels = labels.to(device)
        
#         optimizer.zero_grad()
        
#         outputs = net(images)
#         loss = criterion(outputs, labels)
        
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
    
#     epoch_loss = running_loss / len(train_dataloader)
#     print(f"Epoch {epoch+1} Loss: {epoch_loss}")
    
# torch.save(net.state_dict(), './models/model100.pth')

# model loading


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy}")

# image_path = './data/image/img1.jpg'
# image = Image.open(image_path)
# image = transform_test(image)
# image = torch.unsqueeze(image, 0)
# image = image.to(device)


# with torch.no_grad():
#     output = net(image)

# # 예측 결과 확인
# predicted_label = torch.argmax(output, dim=1)
# print("Predicted Label:", predicted_label.item())