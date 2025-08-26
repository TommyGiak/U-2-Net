import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import pandas as pd
import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SemanticSalObjDataset

from model import SU2NET
from model import SU2NETP

# ------- 1. define loss function --------

ce_loss = nn.CrossEntropyLoss(reduction='mean')

def muti_ce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = ce_loss(d0,torch.argmax(labels_v, dim=1))
	loss1 = ce_loss(d1,torch.argmax(labels_v, dim=1))
	loss2 = ce_loss(d2,torch.argmax(labels_v, dim=1))
	loss3 = ce_loss(d3,torch.argmax(labels_v, dim=1))
	loss4 = ce_loss(d4,torch.argmax(labels_v, dim=1))
	loss5 = ce_loss(d5,torch.argmax(labels_v, dim=1))
	loss6 = ce_loss(d6,torch.argmax(labels_v, dim=1))

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'

data_dir = os.path.join(os.getcwd(), 'train_data_real' + os.sep)
tra_image_dir = []
tra_label_dir = [] 
model_dir = []
for i in range(5):
  tra_image_dir.append(os.path.join(f'train_data_real{i}' + os.sep + 'wound' + os.sep))
  tra_label_dir.append(os.path.join(f'train_data_real{i}' + os.sep + 'semantic_masks' + os.sep))
  model_dir.append(os.path.join(os.getcwd(), 'saved_models', model_name + '_real' + str(i) + os.sep))

  os.makedirs(model_dir[i], exist_ok=True)

tra_image_dir = [] #!!! TAKEN OUT ALL THE REAL FOLDERS!!!

image_ext = '.png'
label_ext = '.png'


epoch_num = 100
batch_size_train = 4
batch_size_val = 1
train_num = 0
val_num = 0

if __name__=='__main__':

    for i, tra_image_dir_i in enumerate(tra_image_dir):

        tra_img_name_list = glob.glob(data_dir + tra_image_dir_i + '*' + image_ext)

        tra_lbl_name_list = []
        for img_path in tra_img_name_list:
            img_name = img_path.split(os.sep)[-1]

            aaa = img_name.split(".")
            bbb = aaa[0:-1]
            imidx = bbb[0]
            for k in range(1,len(bbb)):
                imidx = imidx + "." + bbb[k]

            tra_lbl_name_list.append(data_dir + tra_label_dir[i] + imidx + label_ext)

        print("---")
        print("train images: ", len(tra_img_name_list))
        print("train labels: ", len(tra_lbl_name_list))
        print("---")

        train_num = len(tra_img_name_list)


        salobj_dataset = SemanticSalObjDataset(
            img_name_list=tra_img_name_list,
            lbl_name_list=tra_lbl_name_list,
            transform=transforms.Compose([
                RescaleT(320),
                RandomCrop(288),
                ToTensorLab(flag=0)]))

        salobj_dataloader = DataLoader(salobj_dataset, 
                                        batch_size=batch_size_train, 
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=4,
                                        )

        # ------- 3. define model --------
        # define the net
        if(model_name=='u2net'):
            net = SU2NET(in_ch=3, out_ch=4)
        elif(model_name=='u2netp'):
            net = SU2NETP(in_ch=3, out_ch=4)

        if torch.cuda.is_available():
            net.to('cuda')

        print(f'USING DEVICE: {next(net.parameters()).device}')

        # ------- 4. define optimizer --------
        print("---define optimizer...")
        optimizer = optim.Adam(net.parameters(), lr=0.0004, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1, end_factor=1/10, total_iters=50)

        # ------- 5. training process --------
        print("---start training...")
        ite_num = 0
        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0
        save_frq = 10 # save the model every 2000 iterations
        lossi = []

        for epoch in range(0, epoch_num):
            net.train()

            for k, data in enumerate(salobj_dataloader):
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1

                inputs, labels = data['image'], data['label']

                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs_v, labels_v = Variable(inputs.to('cuda', non_blocking=True), requires_grad=False), Variable(labels.to('cuda', non_blocking=True),
                                                                                                requires_grad=False)
                else:
                    inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

                # y zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
                loss2, loss = muti_ce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

                loss.backward()
                optimizer.step()
                scheduler.step()

                # # print statistics
                running_loss += loss.data.item()
                running_tar_loss += loss2.data.item()
                lossi.append(loss.item())

                # del temporary outputs and loss
                del d0, d1, d2, d3, d4, d5, d6, loss2, loss
            
            
                print("[train set: %d, epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                i, epoch + 1, epoch_num, (k + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if (epoch+1) % save_frq == 0:
                torch.save(net.state_dict(), model_dir[i] + model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (epoch+1, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

        pd.DataFrame(lossi).to_csv(model_dir[i]+'lossi.csv')



### Training with synthetic images

    print('TRAINING WITH SYNTH IMAGES!')

    tra_img_name_list = glob.glob('train_data_synthetic/wounds/' + '*.png') #!!! USING ALL SYNTH

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split('/')[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for k in range(1,len(bbb)):
            imidx = imidx + "." + bbb[k]

        tra_lbl_name_list.append('train_data_synthetic/semantic_masks/' + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SemanticSalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))

    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)


    # ------- 3. define model --------
    # define the net
    if(model_name=='u2net'):
        net = SU2NET(in_ch=3, out_ch=4)
    elif(model_name=='u2netp'):
        net = SU2NETP(in_ch=3, out_ch=4)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.0004, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1, end_factor=1/10, total_iters=50)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 10 # save the model every 2000 iterations
    lossi = []

    for epoch in range(0, epoch_num):
        net.train()

        for k, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_ce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()
            lossi.append(loss.item())

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss
        
        
            print("[train synthetic, epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (k + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
        
        os.makedirs('saved_models/u2net_synth_full/',exist_ok=True)  #!!! USING ALL SYNTH
        if (epoch+1) % save_frq == 0:
            torch.save(net.state_dict(), 'saved_models/u2net_synth_full/' + model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (epoch+1, running_loss / ite_num4val, running_tar_loss / ite_num4val))  #!!! USING ALL SYNTH
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

    pd.DataFrame(lossi).to_csv('saved_models/u2net_synth_full/'+'lossi.csv')  #!!! USING ALL SYNTH