import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision import transforms
import numpy as np

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, list_ids,labels):
        
        self.labels = labels
        self.list_ids = list_ids
        
    def __len__(self):
        
        return len(self.list_ids)
    
    def __getitem__(self, index):

        ID = self.list_ids[index][5]

        if(ID[-4:] == "jpeg" or ID[-4:] == "tiff"):
            
            # read RGB, read mask, read boundries, read normals
            X = read_image("/home/negar/Documents/Datasets/pix3d/Pix3D/"+ID[:-4]+"png")
            Z = read_image("/home/negar/Documents/Datasets/pix3d/Pix3D/crop_mask/"+ID[5:-4]+"png")[1,:,:]
            boundries = read_image("/home/negar/Documents/Datasets/pix3d/Pix3D/boundries/"+ID[5:-4]+"png")
            normals = read_image("/home/negar/Documents/Datasets/pix3d/Pix3D/normal/"+ID[5:-4]+"png")
            
        else:
            
            # read RGB, read mask, read boundries, read normals
            X = read_image("/home/negar/Documents/Datasets/pix3d/Pix3D/"+ID[:-3]+"png")
            Z = read_image("/home/negar/Documents/Datasets/pix3d/Pix3D/crop_mask/"+ID[5:-3]+"png")[1,:,:]
            boundries = read_image("/home/negar/Documents/Datasets/pix3d/Pix3D/boundries/"+ID[5:-3]+"png")
            normals = read_image("/home/negar/Documents/Datasets/pix3d/Pix3D/normal/"+ID[5:-3]+"png")
            
        #labels    
        y = torch.from_numpy(self.labels[index])
        return torch.cat((X.float(),Z.reshape(1,224,224).float() ,boundries.float(),normals.float())), y      



class Net_combined(nn.Module):

    def __init__(self, rgb, mask, bound, normal):
        
        super().__init__()
        self.conv1 = nn.Conv2d(3*rgb + mask + bound + 3*normal, 64, 7)
        self.bn1 = nn.BatchNorm2d(64)
        #self.conv1_2 = nn.Conv2d(64, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 256, 7)
        self.bn2 = nn.BatchNorm2d(256)
        #self.conv2_2 = nn.Conv2d(128, 128, 3)

        self.conv3 = nn.Conv2d(256, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)
        #self.conv3_2 = nn.Conv2d(256, 256, 3)

        self.conv4 = nn.Conv2d(256, 8, 3)
        self.bn4 = nn.BatchNorm2d(8)
        #self.conv4_2 = nn.Conv2d(512, 512, 3)
        
        self.fc = nn.Linear(8*121, 128)
        self.fc1 = nn.Linear(128, 8)
        self.fc2 = nn.Linear(128, 8)
        self.fc3 = nn.Linear(128, 8)
        
        #self.rlu = nn.LeakyReLU(0.1)
        #self.drp = nn.Dropout(p=0.5)
        #self.m = nn.Softmax2d()
        #self.m = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc(x))
        w = F.relu(self.fc1(x))
        y = F.relu(self.fc2(x))
        z = F.relu(self.fc3(x))
        
        return w,y,z


class Net_separate(nn.Module):

    def __init__(self, rgb, mask, bound, normal,device):
        
        super().__init__()
        self.rgb =rgb
        self.mask = mask
        self.bound = bound
        self.normal = normal
        self.device = device
        
        # conv layers for RGB
        self.conv1_rgb = nn.Conv2d(3, 32, 7)
        self.bn1_rgb = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2_rgb = nn.Conv2d(32, 64, 7)
        self.bn2_rgb = nn.BatchNorm2d(64)

        #self.conv3_rgb = nn.Conv2d(256, 256, 3)
        #self.bn3_rgb = nn.BatchNorm2d(256)

        self.conv4_rgb = nn.Conv2d(64, 8, 3)
        self.bn4_rgb = nn.BatchNorm2d(8)
        self.fc_rgb = nn.Linear(4608, 128)

        
           
        # conv layers for mask
        self.conv1_mask = nn.Conv2d(1, 32, 7)
        self.bn1_mask = nn.BatchNorm2d(32)
        
        self.conv2_mask = nn.Conv2d(32, 64, 7)
        self.bn2_mask = nn.BatchNorm2d(64)

        #self.conv3_mask = nn.Conv2d(256, 256, 3)
        #self.bn3_mask = nn.BatchNorm2d(256)

        self.conv4_mask = nn.Conv2d(64, 8, 3)
        self.bn4_mask = nn.BatchNorm2d(8)
        self.fc_mask = nn.Linear(4608, 128)

        
        
        # conv layers for boundries
        self.conv1_bound = nn.Conv2d(1, 32, 7)
        self.bn1_bound = nn.BatchNorm2d(32)
        
        self.conv2_bound = nn.Conv2d(32, 64, 7)
        self.bn2_bound = nn.BatchNorm2d(64)

        #self.conv3_bound = nn.Conv2d(256, 256, 3)
        #self.bn3_bound = nn.BatchNorm2d(256)

        self.conv4_bound = nn.Conv2d(64, 8, 3)
        self.bn4_bound = nn.BatchNorm2d(8)
        self.fc_bound = nn.Linear(4608, 128)

        
        # conv layers for normals
        self.conv1_normal = nn.Conv2d(3, 32, 7)
        self.bn1_normal = nn.BatchNorm2d(32)
        
        self.conv2_normal = nn.Conv2d(32, 64, 7)
        self.bn2_normal = nn.BatchNorm2d(64)

        #self.conv3_normal = nn.Conv2d(256, 256, 3)
        #self.bn3_normal = nn.BatchNorm2d(256)

        self.conv4_normal = nn.Conv2d(64, 8, 3)
        self.bn4_normal = nn.BatchNorm2d(8)
        self.fc_normal = nn.Linear(4608, 128)

        
        
        self.fc1 = nn.Linear((rgb + mask + bound + normal)* 128, 8)
        self.fc2 = nn.Linear((rgb + mask + bound + normal)* 128, 8)
        self.fc3 = nn.Linear((rgb + mask + bound + normal)* 128, 8)


    def forward(self, input_x):
        
        x1 = torch.Tensor().to(self.device)
        x2 = torch.Tensor().to(self.device)
        x3 = torch.Tensor().to(self.device)
        x4 = torch.Tensor().to(self.device)
        
        
        if ( self.rgb ==1 ) :
            x = self.pool(F.relu(self.bn1_rgb(self.conv1_rgb(input_x[:,0:3,:,:]))))
            x = self.pool(F.relu(self.bn2_rgb(self.conv2_rgb(x))))
            #x = self.pool(F.relu(self.bn3_rgb(self.conv3_rgb(x))))
            x = self.pool(F.relu(self.bn4_rgb(self.conv4_rgb(x))))
            x = x.view(x.shape[0], -1)
            x1 = F.relu(self.fc_rgb(x))
        
        if ( self.mask ==1 ) :
            x = self.pool(F.relu(self.bn1_mask(self.conv1_mask(input_x[:,3:4,:,:]))))
            x = self.pool(F.relu(self.bn2_mask(self.conv2_mask(x))))
            #x = self.pool(F.relu(self.bn3_mask(self.conv3_mask(x))))
            x = self.pool(F.relu(self.bn4_mask(self.conv4_mask(x))))
            x = x.view(x.shape[0], -1)
            x2 = F.relu(self.fc_mask(x))
            
        if ( self.bound ==1 ) :
            x = self.pool(F.relu(self.bn1_bound(self.conv1_bound(input_x[:,4:5,:,:]))))
            x = self.pool(F.relu(self.bn2_bound(self.conv2_bound(x))))
            #x = self.pool(F.relu(self.bn3_bound(self.conv3_bound(x))))
            x = self.pool(F.relu(self.bn4_bound(self.conv4_bound(x))))
            x = x.view(x.shape[0], -1)
            x3 = F.relu(self.fc_bound(x))
            
        if ( self.normal == 1 ) :
            x = self.pool(F.relu(self.bn1_normal(self.conv1_normal(input_x[:,5:8,:,:]))))
            x = self.pool(F.relu(self.bn2_normal(self.conv2_normal(x))))
            #x = self.pool(F.relu(self.bn3_normal(self.conv3_normal(x))))
            x = self.pool(F.relu(self.bn4_normal(self.conv4_normal(x))))
            x = x.view(x.shape[0], -1)
            x4 = F.relu(self.fc_normal(x))
        
        
        final_feature_map = torch.cat((torch.flatten(x1),torch.flatten(x2),torch.flatten(x3),torch.flatten(x4)))
        reshape_final_feature_map = torch.reshape(final_feature_map,(x.shape[0], -1))

        w = F.relu(self.fc1(reshape_final_feature_map))
        y = F.relu(self.fc2(reshape_final_feature_map))
        z = F.relu(self.fc3(reshape_final_feature_map))

        return w,y,z
    
    
    