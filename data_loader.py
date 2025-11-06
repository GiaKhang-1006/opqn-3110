import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.io
import os
import cv2
#import dlib
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as TF

#detector = dlib.get_frontal_face_detector()  # Global dlib detector for align

def get_datasets_transform(dataset, data_dir="/kaggle/input/facescrub-edgeface-0710-1", cross_eval=False, backbone='resnet'):
    to_tensor = transforms.ToTensor()

    # Define paths for FaceScrub processed data
    if dataset == "facescrub":
        train_path = os.path.join(data_dir, "facescrub", "train", "actors")
        test_path = os.path.join(data_dir, "facescrub", "test", "actors")
    elif dataset == "vggface2":
        if cross_eval:
            train_path = os.path.join(data_dir, "vggface2", "cross_train")
            test_path = os.path.join(data_dir, "vggface2", "cross_test")
        else:
            train_path = os.path.join(data_dir, "vggface2", "train")
            test_path = os.path.join(data_dir, "vggface2", "test")
    else:
        train_path = os.path.join(data_dir, dataset, "train")
        test_path = os.path.join(data_dir, dataset, "test")

    # Load datasets with debug print
    trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
    testset = datasets.ImageFolder(root=test_path, transform=to_tensor)
    print(f"Train path: {train_path}")  # Debug
    print(f"Test path: {test_path}")    # Debug

    # Align only for EdgeFace if dataset not pre-aligned (FaceScrub is pre-aligned 112x112)
    align_transform = nn.Identity()  # Skip align since dataset is pre-aligned
    # Uncomment below if you want to enable alignment for EdgeFace
    # align_transform = transforms.Lambda(align_face) if backbone == 'edgeface' else nn.Identity()

    # Normalize and resize conditional
    # if backbone == 'edgeface':
    #     # norm_mean = [0.618, 0.465, 0.393]
    #     # norm_std = [0.238, 0.202, 0.190]
    #     norm_mean = (0.5, 0.5, 0.5) #Norm [-1 1] thay vì. [0 1]
    #     norm_std = (0.5, 0.5, 0.5)
    #     resize_crop_size = 120
    #     crop_size = 112
        
    # Normalize and resize conditional
    if backbone == 'edgeface':
        # TỰ ĐỘNG PHÁT HIỆN DATA 32x32 QUA ĐƯỜNG DẪN
        if '32x32' in data_dir.lower() or '32' in data_dir.lower():
            # DÙNG CHO DATASET 32x32 → CHỈ ĐỔI KÍCH THƯỚC
            norm_mean = (0.5, 0.5, 0.5)  # GIỮ NGUYÊN [-1, 1]
            norm_std  = (0.5, 0.5, 0.5)
            resize_crop_size = 35
            crop_size = 32
            print("EdgeFace: Phát hiện dataset 32x32 → Resize(35) + Crop(32)")
        else:
            # MẶC ĐỊNH 112x112
            norm_mean = (0.5, 0.5, 0.5)
            norm_std  = (0.5, 0.5, 0.5)
            resize_crop_size = 120
            crop_size = 112
            print("EdgeFace: Dùng Resize(120) + Crop(112)")

    else:  # resnet (gốc OPQN) ← ĐÚNG VỊ TRÍ
        if dataset == "vggface2" or cross_eval:
            norm_mean = (0.5, 0.5, 0.5)
            norm_std  = (0.5, 0.5, 0.5)
            resize_crop_size = 120
            crop_size = 112
        else:  # facescrub
            norm_mean = [0.639, 0.479, 0.404]
            norm_std  = [0.216, 0.183, 0.171]
            resize_crop_size = 35
            crop_size = 32

    # Transforms
    if cross_eval:
        transform_train = nn.Sequential(
            align_transform,
            transforms.Resize(resize_crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(norm_mean, norm_std),
        )
        transform_test = transform_train
    else:
        if dataset == "vggface2":
            transform_train = nn.Sequential(
                align_transform,
                transforms.Resize(resize_crop_size),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),  # Thêm giống EdgeFace
                transforms.GaussianBlur(kernel_size=3),  # Thêm giống EdgeFace
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(norm_mean, norm_std),
            )
            transform_test = nn.Sequential(
                align_transform,
                transforms.Resize(resize_crop_size),
                transforms.CenterCrop(crop_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(norm_mean, norm_std),
            )
        else:
            transform_train = nn.Sequential(
                align_transform,
                transforms.Resize(resize_crop_size),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),  # Thêm giống EdgeFace
                transforms.GaussianBlur(kernel_size=3),  # Thêm giống EdgeFace
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(norm_mean, norm_std),
            )
            transform_test = nn.Sequential(
                align_transform,
                transforms.Resize(resize_crop_size),
                transforms.CenterCrop(crop_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(norm_mean, norm_std),
            )

    return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}