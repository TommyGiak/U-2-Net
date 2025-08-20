import os
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as standard_transforms

from tqdm.auto import tqdm
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Device: {device}')

# creating list of images and dataloaders

complete_data_folder = 'train_data_synthetic/wounds/'
fruci_folder = 'train_data_synthetic/validated_fruci/'
label_folder = 'train_data_synthetic/semantic_masks/'

complete_data_list = glob.glob(complete_data_folder+'*.png')
fruci_list = glob.glob(fruci_folder+'*.png')
label_list = glob.glob(label_folder+'*.png')

test_img_list = complete_data_list.copy()
test_lbl_list = label_list.copy()

print(f'Found {len(complete_data_list)} images in {complete_data_folder}')
print(f'Found {len(fruci_list)} images in {fruci_folder}')
print(f'Found {len(label_list)} images in {label_folder}')

print('starting filtering test images with train images...')
for path in tqdm(fruci_list):
  img = path.split('/')[-1]
  for path_test in complete_data_list:
    img_test = path_test.split('/')[-1]
    if img == img_test:
      test_img_list.remove(path_test)
      path_lbl = label_folder+img_test
      test_lbl_list.remove(path_lbl)

print(f'Final number of images: {len(test_img_list)}')
print(f'Final number of lables: {len(test_lbl_list)}')


salobj_dataset = SemanticSalObjDataset(
  img_name_list=test_img_list,
  lbl_name_list=test_lbl_list,
  transform=transforms.Compose([
    RescaleT(320),
    RandomCrop(288),
    ToTensorLab(flag=0)]))

salobj_dataloader = DataLoader(salobj_dataset, 
                                batch_size=1, 
                                shuffle=False,
                                pin_memory=True,
                                #num_workers=8,
                                )

# importing the model

model = SU2NET(in_ch=3, out_ch=4).to(device)
model.eval()

models_paths = glob.glob(f'saved_models/u2net_synth/u2net_bce_itr_100*')
assert len(models_paths)==1

model.load_state_dict(torch.load(models_paths[0], map_location=device))

EPS = 1e-6

def compute_metrics(one_hot, lable, layers, d):
  if len(layers)>1:
    pred = one_hot[layers[0]]
    gt = lable[layers[0]]
    for l in layers[1:]:
      pred += one_hot[l]
      gt += lable[l]
  else:
    pred = one_hot[layers]
    gt = lable[layers]

  tp = (pred*gt).sum()
  fp = (pred*torch.logical_not(gt)).sum()
  tn = (torch.logical_not(pred)*torch.logical_not(gt)).sum()
  fn = (torch.logical_not(pred)*gt).sum()
  tot = tp+fp+tn+fn

  assert tot == torch.ones_like(pred).sum()

  d['precision'].append(((tp+EPS)/(tp+fp+EPS)).item())
  d['recall'].append(((tp+EPS)/(tp+fn+EPS)).item())
  d['accuracy'].append(((tp+tn+EPS)/(tot+EPS)).item())
  d['dice'].append(((2*tp+EPS)/(2*tp+fp+fn+EPS)).item())
  d['iou'].append(((tp+EPS)/(tp+fp+fn+EPS)).item())

  return d

  # metrics computation
from tqdm.auto import tqdm

wound_metrics = {'image':[], 'precision':[], 'recall':[], 'accuracy':[], 'dice':[], 'iou':[]}
body_wound_marker_metrics = {'image':[], 'precision':[], 'recall':[], 'accuracy':[], 'dice':[], 'iou':[]}
marker_metrics = {'image':[], 'precision':[], 'recall':[], 'accuracy':[], 'dice':[], 'iou':[]}
body_marker_no_wound_metrics = {'image':[], 'precision':[], 'recall':[], 'accuracy':[], 'dice':[], 'iou':[]}
body_only_metrics = {'image':[], 'precision':[], 'recall':[], 'accuracy':[], 'dice':[], 'iou':[]}

with torch.no_grad():
  for data in tqdm(salobj_dataloader):
    inputs, labels = torch.tensor(data['image'], device=device, dtype=torch.float32), torch.tensor(data['label'], device=device, dtype=torch.float32)

    d1,d2,d3,d4,d5,d6,d7= model(inputs)

    # normalization
    dim = 1

    # adding image path
    wound_metrics['image'].append(test_img_list[data['imidx']])
    body_wound_marker_metrics['image'].append(test_img_list[data['imidx']])
    marker_metrics['image'].append(test_img_list[data['imidx']])
    body_marker_no_wound_metrics['image'].append(test_img_list[data['imidx']])
    body_only_metrics['image'].append(test_img_list[data['imidx']])


    indices = torch.argmax(d1, dim=dim)
    one_hot = torch.nn.functional.one_hot(indices, num_classes=d1.size(dim)).bool().permute((0,3,1,2)).squeeze()
    
    indices_lbl = torch.argmax(labels, dim=dim)
    labels = torch.nn.functional.one_hot(indices_lbl, num_classes=d1.size(dim)).bool().permute((0,3,1,2)).squeeze()


    wound_metrics = compute_metrics(one_hot, labels, layers=[3], d=wound_metrics)
    body_wound_marker_metrics = compute_metrics(one_hot, labels, layers=[1,2,3], d=body_wound_marker_metrics)
    marker_metrics = compute_metrics(one_hot, labels, layers=[2], d=marker_metrics)
    body_marker_no_wound_metrics = compute_metrics(one_hot, labels, layers=[1,2], d=body_marker_no_wound_metrics)
    body_only_metrics = compute_metrics(one_hot, labels, layers=[1], d=body_only_metrics)


os.makedirs('metrics/', exist_ok=True)
os.makedirs(f'metrics/model_synth_synth/', exist_ok=True)
pd.DataFrame(wound_metrics).to_csv(f'metrics/model_synth_synth/wound.csv')
pd.DataFrame(body_wound_marker_metrics).to_csv(f'metrics/model_synth_synth/body_wound_marker.csv')
pd.DataFrame(marker_metrics).to_csv(f'metrics/model_synth_synth/marker.csv')
pd.DataFrame(body_marker_no_wound_metrics).to_csv(f'metrics/model_synth_synth/body_marker_no_wound.csv')
pd.DataFrame(body_only_metrics).to_csv(f'metrics/model_synth_synth/body_only.csv')
