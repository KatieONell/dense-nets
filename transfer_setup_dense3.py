# Make a deep ReLU network and train it to label images

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import shelve
import math

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels) #subtract out average / div s.d.
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,stride=1,padding=1,bias=False)
        self.conv1_drop = nn.Dropout2d(p=0.1) #regularization
    def forward(self, x):
        out = self.conv1_drop(F.relu(self.conv1(self.bn1(x.float()))))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self,nChannels,nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=3,stride=1,padding=1,bias=False)
        self.conv1_drop = nn.Dropout2d(p=0.1)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self,x):
        out = self.pool(self.conv1_drop(F.relu(self.conv1(self.bn1(x.float())))))
        return out

class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses):
        super(DenseNet, self).__init__()
        nDenseBlocks = (depth-4) // 3
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,bias=True)
        print(nChannels)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        print(nChannels)
        self.trans1 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        print(nChannels)
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        print(nChannels)
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        print(nChannels)
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate
        print(nChannels)
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)
        print(nChannels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
            #
    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)
        #
    def forward(self, x):
        out = self.conv1(x.float())
        out = self.trans1(self.dense1(out.float()))
        out = self.trans2(self.dense2(out.float()))
        out = self.dense3(out.float())
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out.float())), 8))
        out = F.log_softmax(self.fc(out.float()))
        return out

#growthRate=32
#depth=13
#reduction=0.5
#nClasses=7
net = DenseNet(32, 13, 0.5, 7)
net.cuda()

#now read back in old model
savedModel=torch.load('/mindhive/saxelab3/anzellotti/deepnet/expressions_identity/fer2013/fer2013/denseV2_32_500_k3_katietest/199.ckpt')
#savedModel=torch.load('/mindhive/saxelab3/anzellotti/deepnet/expressions_identity/fer2013/fer2013/denseV2_32_500_k3_katietest_transfer_identity/499.ckpt')
state_dict=savedModel['state_dict']
optimizer=savedModel['optimizer']
epoch=savedModel['epoch']
net.load_state_dict(state_dict)

#### now introduce transfer layer stuff

class convNet(nn.Module):
    #constructor
    def __init__(self,resnet,myNet):
        super(convNet, self).__init__()
        #defining layers in convnet
        self.resnet=resnet
        myNet=myNet
        self.resnet.conv1.requires_grad=False
        self.resnet.dense1.requires_grad=False
        self.resnet.trans1.requires_grad=False
        self.resnet.dense2.requires_grad=False
        self.resnet.trans2.requires_grad=False
        self.resnet.dense3.requires_grad=False
        #self.resnet.bn1 = nn.BatchNorm2d(32)
        self.resnet.bn1 = nn.BatchNorm2d(184)
        self.resnet.bn1.requires_grad=False #does this do anything?
        self.resnet.fc=myNet 
        #self.resnet=nn.Sequential(*[self.resnet.conv1,self.resnet.dense1,self.resnet.fc]) 
    def forward(self, x):
        out = self.resnet.conv1(x.float())
        out = self.resnet.trans1(self.resnet.dense1(out.float()))
        out = self.resnet.trans2(self.resnet.dense2(out.float()))
        out = self.resnet.dense3(out.float())
        out = torch.squeeze(F.avg_pool2d(F.relu(self.resnet.bn1(out.float())), 8))
        #out = out.view(-1,36*64) #the 64 comes from the 2*growth rate term; 36 from ??
        out=out.view(-1,9*184)
        out = F.log_softmax(self.resnet.fc(out.float()))
        return out

#growthRate=32
#depth=13
#reduction=0.5
#nClasses=7
#myNet=nn.Linear(36*64, 70)
myNet=nn.Linear(9*184, 70)
newNet=convNet(net,myNet)
newNet.cuda()


#bring in dataset
# Import data path
import os
import glob
from random import *
import csv
dataPath = r'/mindhive/saxelab3/anzellotti/deepnet/expressions_identity/fer2013/fer2013/denseV2_32_500_k3_katietest/identity_transfer_data/holdout_neutral_with45.csv'
#dataPath = '/mindhive/saxelab3/anzellotti/deepnet/expressions_identity/fer2013/fer2013/denseV2_32_500_k3_katietest/stimuli_test_cropped_NEW_5.csv'
#dataPath = r'/mindhive/saxelab3/anzellotti/deepnet/expressions_identity/fer2013/fer2013/denseV2_32_500_k3_katietest/transfer_identity.csv'

# Implement the data loader.
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils

class KarolinskaDataset(Dataset):
    def __init__(self, csv_file, usage, transform=None):
        dataInfo_temp = pd.read_csv(csv_file)
        # take only the part of the file that's relevant for the current usage
        self.dataInfo = (dataInfo_temp[dataInfo_temp.Usage == usage]).reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.dataInfo)
    def __getitem__(self, idx):
        image = np.reshape(np.array(np.fromstring(self.dataInfo.ix[idx, 1], np.uint8, sep=' ')),(388,388)) # reshape image
        label = self.dataInfo.ix[idx, 0]   
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size , self.output_size* w / w
            else:
                new_h, new_w = self.output_size* h / w, self.output_size 
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'label': label}

class Flip(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.randint(0,1)>0:
            image = np.fliplr(image)
        return {'image': image, 'label': label}

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = torch.from_numpy(image)
        return {'image': torch.unsqueeze(image,0), # add 
                'label': torch.from_numpy(np.array([label]))}


#create trainloader, and train new outer layer
karolinska_train_transformed = KarolinskaDataset(csv_file=dataPath,
                                           usage='Training',
                                           transform=transforms.Compose([
                                               Resize(48),
                                               Flip(),
                                               ToTensor(),
                                           ]))

trainloader = DataLoader(karolinska_train_transformed, batch_size=32,
                        shuffle=True, num_workers=2,pin_memory=False)

# Train the network
import torch.optim as optim
from torch.autograd import Variable
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9) # typical
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,weight_decay=0.001) # with l2 regularization
# optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.9) # for 0.5 dropout
# optimizer = optim.Adagrad(net.parameters(), lr=0.01, lr_decay=0, weight_decay=0)

def save_model(net,optim,ckpt_fname):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optim},
        ckpt_fname)

def adjust_learning_rate(optimizer, epoch):
    lr = 0.1*(0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

newNet.train()
save_freq = 50
save_dir = '/mindhive/saxelab3/anzellotti/deepnet/expressions_identity/fer2013/fer2013/denseV2_32_500_k3_katietest_transfer_holdout_neutral_DENSE2'
os.mkdir(save_dir)
loss_memory = []
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
for epoch in range(200):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        adjust_learning_rate(optimizer,epoch)
        # get the inputs
        images = data['image']
        labels = data['label']
        # wrap them in Variable
        tmp = []
        tmp = torch.squeeze(labels.long())
        images, labels = Variable(images.cuda()),  Variable(tmp.cuda())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = newNet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.data[0]
        if i % 50 == 49:    # print every 150 mini-batches
            loss_memory.append(running_loss/50)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
#            scheduler.step(running_loss / 50)
            running_loss = 0.0
    if epoch % save_freq == save_freq-1: 
        save_model(newNet, optimizer, os.path.join(save_dir, '%03d.ckpt' % epoch)) 


#create dataset and data loader!
#karolinska_train_transformed = None
#trainloader = None
karolinska_test_transformed = KarolinskaDataset(csv_file=dataPath,
                                            usage='PublicTest',
                                            transform=transforms.Compose([
                                               Resize(48),
                                               Flip(),
                                               ToTensor(),
                                           ]))
testloader = DataLoader(karolinska_test_transformed, batch_size=32,
                        shuffle=True, num_workers=2,pin_memory=False)

#do the test! :)
from torch.autograd import Variable
newNet.eval()
score = []
for i, data in enumerate(testloader):
        # get the inputs
        images = data['image']
        labels = data['label']
        # wrap them in Variable
        tmp = []
        tmp = torch.squeeze(labels.long())
        images, labels = Variable(images.cuda()),  Variable(tmp.cuda())
        # forward + backward + optimize
        outputs = newNet(images)
        outputs_numpy = outputs.cpu().data.numpy()
        outputs_argmax = np.argmax(outputs_numpy,axis=1)
        labels_numpy = labels.cpu().data.numpy()
        score = np.concatenate((score,(labels_numpy==outputs_argmax).astype(int)),axis=0)

meanAccuracy = sum(score)/len(score)
print(meanAccuracy)