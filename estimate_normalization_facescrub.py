
import os
import glob
import torch
import torchvision.io
import numpy as np

root_dir = '/kaggle/input/facescrub-edgeface-0710-1/facescrub'
train_files = glob.glob(os.path.join(root_dir, 'train', 'actors', '**', '*.jpeg'), recursive=True) + glob.glob(os.path.join(root_dir, 'train', 'actors', '**', '*.jpg'), recursive=True)
test_files = glob.glob(os.path.join(root_dir, 'test', 'actors', '**', '*.jpeg'), recursive=True) + glob.glob(os.path.join(root_dir, 'test', 'actors', '**', '*.jpg'), recursive=True)
files = train_files + test_files

print('Total files:', len(files))

mean_r, mean_g, mean_b = [], [], []
std_r, std_g, std_b = [], [], []
for f in files:
    img = torchvision.io.read_image(f).float() / 255
    mean_r.append(img[0].mean().item())
    mean_g.append(img[1].mean().item())
    mean_b.append(img[2].mean().item())
    std_r.append(img[0].std().item())
    std_g.append(img[1].std().item())
    std_b.append(img[2].std().item())

print('Mean RGB:', [np.mean(mean_r), np.mean(mean_g), np.mean(mean_b)])
print('Std RGB:', [np.mean(std_r), np.mean(std_g), np.mean(std_b)])
