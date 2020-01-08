import itertools

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
# cuda = torch.device(0)
batch_size = 64
transform = transforms.Compose(
    [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
trn_dataset = datasets.CIFAR10('./cifar_data/',
                               download=True,
                               train=True,
                               transform=transform)

trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)  # s = TRUE

val_dataset = datasets.CIFAR10("./cifar_data/",
                               download=True,
                               train=False,
                               transform=transform)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dense1 = nn.Linear(6400, 4096)  # 완전연결 1차원 배열로?
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        x = torch.flatten(x, 1)
        # x = x.view(-1,256*5*5)
        x = self.dense3(self.drop2(F.relu(self.dense2(self.drop1(F.relu(self.dense1(x)))))))
        return x


# cnn = CNNClassifier()

cnn = CNNClassifier().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.0001)
train_start = time.time()

for epoch in range(10):
    trn_loss = 0.0
    start = time.time()
    for i, data in enumerate(trn_loader):
        x, label = data

        x = x.cuda()
        label = label.cuda()
        x = Variable(x)
        label = Variable(label)

        optimizer.zero_grad()
        model_output = cnn(x)
        loss = criterion(model_output, label)
        loss.backward()
        optimizer.step()

        trn_loss += loss.data
        # loss.item() # loss 스칼라 값
        del loss
        del model_output

        if (i + 1) % 100 == 0:
            end = time.time()
            print ('[epoch %d,imgs %5d] loss: %.7f time : %.3f s' % (
            epoch + 1, (i + 1) * 64, trn_loss / 100, (end - start)))
            start = time.time()
            trn_loss = 0
train_end = time.time()
print("train end, AlexNet batch_size : 64 train time : %.3f s" % (train_end - train_start))

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