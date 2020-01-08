import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import time

batch_size=32
transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
trn_dataset = datasets.CIFAR10('../CIFAR10_data/',
                             download=True,
                             train=True,
                             transform=transform) 

trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

val_dataset = datasets.CIFAR10("../CIFAR10_data/", 
                             download=True,
                             train=False,
                             transform= transform)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)
#########################################################
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
      self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
      self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
      self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1)
      self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  
      self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1) 
      self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1)       
      self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1)
      self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1)
      self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=1,padding=1)
      self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1)
      self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1)
      self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1)
      self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1)
      self.conv14 = nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1)
      self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1)
      self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1)
      self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
            #여기부터 
      self.dense1 = nn.Linear(25088, 4096) 
      self.drop1 = nn.Dropout(0.5)
      self.dense2 = nn.Linear(4096, 4096) 
      self.drop2 = nn.Dropout(0.5)
      self.dense3 = nn.Linear(4096, 10)

    def forward(self, x):
      x=self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
      x=self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
      x=self.pool3(F.relu(self.conv8(F.relu(self.conv7((F.relu(self.conv6(F.relu(self.conv5(x))))))))))
      x=self.pool4(F.relu(self.conv12(F.relu(self.conv11((F.relu(self.conv10(F.relu(self.conv9(x))))))))))
      x=self.pool5(F.relu(self.conv16(F.relu(self.conv15((F.relu(self.conv14(F.relu(self.conv13(x))))))))))
      x = torch.flatten(x, 1)
      x=self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
      return x

cnn = Net().cuda()
#print(cnn)
criterion = nn.CrossEntropyLoss() # 손실함수
optimizer = optim.Adam(cnn.parameters(), lr=0.01) # 최적화 정의 0.08은 큰
train_start = time.time()
# 손실 함수와 Optimizer를 생성. SGD 생성자에 model.parameters()를 호출하면 모델의 멤버인 2개의 nn.Linear 모듈의 학습 가능한 매개변수들이 포함됨

for epoch in range(3):
    trn_loss = 0.0 #
    start = time.time()
    for i, data in enumerate(trn_loader):
        x, x_labels = data # x.size() = [batch, channel, x, y]

        x = x.cuda()
        x_labels = x_labels.cuda()
        x=Variable(x)
        x_labels=Variable(x_labels)

        optimizer.zero_grad() 
        pred = cnn(x)
        loss = criterion(pred, x_labels)
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()

        #trn_loss += loss.data
        #item() #
        del loss
        del pred

        if (i+1) % 100 == 0:
          end=time.time()
          print ('[epoch %d,imgs %5d] loss: %.7f time : %.3f s'%(epoch+1,(i+1)*32,trn_loss/100, (end - start)))
          start=time.time()
          trn_loss = 0
train_end=time.time()
print("train end, AlexNet batch_size : 64 train time : %.3f s"%(train_end - train_start))

cnn.eval()
correct=0
total=0
for data in val_loader:
    images,labels=data
    images=images.cuda()
    labels=labels.cuda()
    outputs=cnn(Variable(images))
    _,predicted=torch.max(outputs,1)
    total+=labels.size(0)
    correct+=(predicted==labels).sum()
print('Accuracy of the network on the %d test images: %d %%' % (total , 100 * correct / total))