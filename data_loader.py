import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.io
import os
import cv2
import dlib
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as TF

detector = dlib.get_frontal_face_detector()

def get_datasets_transform(dataset, data_dir="/kaggle/input/facescrub-edgeface-32x32", cross_eval=False, backbone='resnet'):
    # Auto detect Kaggle
    if 'kaggle' in os.environ.get('PWD', ''):
        if dataset == 'facescrub':
            base_path = os.path.join(data_dir, 'facescrub')
        else:
            base_path = data_dir
    else:
        base_path = data_dir

    # Define paths
    if dataset == 'facescrub':
        train_path = os.path.join(base_path, 'train', 'actors')
        test_path = os.path.join(base_path, 'test', 'actors')
        if not os.path.exists(train_path):
            train_path = os.path.join(base_path, 'train')
        if not os.path.exists(test_path):
            test_path = os.path.join(base_path, 'test')
    else:
        train_path = os.path.join(base_path, 'train')
        test_path = os.path.join(base_path, 'test')

    print(f"Train path: {train_path}, Test path: {test_path}")

    # Load datasets with dummy transform first
    trainset = datasets.ImageFolder(root=train_path, transform=transforms.ToTensor())
    testset = datasets.ImageFolder(root=test_path, transform=transforms.ToTensor())

    # Detect size from data_dir
    is_32x32 = '32x32' in data_dir.lower() or '32' in data_dir.lower()

    # Align transform (skip for pre-aligned data)
    align_transform = nn.Identity()

    # Normalize & size
    if backbone == 'edgeface':
        if is_32x32:
            norm_mean = (0.5, 0.5, 0.5)
            norm_std  = (0.5, 0.5, 0.5)
            resize_size = 35
            crop_size = 32
        else:
            norm_mean = (0.5, 0.5, 0.5)
            norm_std  = (0.5, 0.5, 0.5)
            resize_size = 120
            crop_size = 112
    else:  # resnet
        if dataset == "vggface2" or cross_eval:
            norm_mean = (0.5, 0.5, 0.5)
            norm_std  = (0.5, 0.5, 0.5)
            resize_size = 120
            crop_size = 112
        else:
            norm_mean = [0.639, 0.479, 0.404]
            norm_std  = [0.216, 0.183, 0.171]
            resize_size = 35
            crop_size = 32

    # FINAL TRANSFORMS
    if cross_eval:
        transform_train = transforms.Compose([
            align_transform,
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
        transform_test = transform_train
    else:
        transform_train = transforms.Compose([
            align_transform,
            transforms.Resize(resize_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
        transform_test = transforms.Compose([
            align_transform,
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

    # Apply transforms to datasets
    trainset.transform = transform_train
    testset.transform = transform_test

    return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}









#Lỗi transform_train không có augmentation
# import torch
# import torchvision.transforms as transforms
# from torchvision import datasets
# import torchvision.io
# import os
# import cv2
# import dlib
# import numpy as np
# import torch.nn as nn
# import torchvision.transforms.functional as TF

# detector = dlib.get_frontal_face_detector()  # Global dlib detector for align

# def get_datasets_transform(dataset, data_dir="/kaggle/input/facescrub-edgeface-0710-1", cross_eval=False, backbone='resnet'):
#     to_tensor = transforms.ToTensor()

#     # Auto detect Kaggle and use specific paths
#     if 'kaggle' in os.environ.get('PWD', ''):
#         if dataset == 'facescrub':
#             base_path = os.path.join(data_dir, 'facescrub')  # e.g., /kaggle/input/facescrub-0210-3/facescrub
#         else:
#             base_path = data_dir  # Fallback for other datasets
#     else:
#         base_path = data_dir  # Local environment

#     # Define paths with folder existence check
#     if dataset == 'facescrub':
#         train_path = os.path.join(base_path, 'train', 'actors')
#         test_path = os.path.join(base_path, 'test', 'actors')
#         if not os.path.exists(train_path):
#             train_path = os.path.join(base_path, 'train')  # Fallback
#         if not os.path.exists(test_path):
#             test_path = os.path.join(base_path, 'test')  # Fallback
#     elif dataset == 'vggface2':
#         if cross_eval:
#             train_path = os.path.join(base_path, 'cross_train') if os.path.exists(os.path.join(base_path, 'cross_train')) else os.path.join(base_path, 'train')
#             test_path = os.path.join(base_path, 'cross_test') if os.path.exists(os.path.join(base_path, 'cross_test')) else os.path.join(base_path, 'test')
#         else:
#             train_path = os.path.join(base_path, 'train')
#             test_path = os.path.join(base_path, 'test')
#     else:
#         train_path = os.path.join(base_path, 'train')
#         test_path = os.path.join(base_path, 'test')
#         if not os.path.exists(train_path):
#             train_path = base_path  # Fallback
#         if not os.path.exists(test_path):
#             test_path = base_path  # Fallback

#     # Debug print
#     print(f"Dataset: {dataset}, Cross-eval: {cross_eval}, Backbone: {backbone}")
#     print(f"Train path: {train_path}, Test path: {test_path}")

#     trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
#     testset = datasets.ImageFolder(root=test_path, transform=to_tensor)

#     # sample_image_path = "/kaggle/input/facescrub-edgeface-0710-1/facescrub/train/actors/Aaron_Eckhart/Aaron_Eckhart_1_1.jpeg"
#     # sample_image = torchvision.io.read_image(sample_image_path)
#     # transformed = transform_train(sample_image)
#     # print("Sample transformed image shape:", transformed.shape, "Mean:", transformed.mean(), "Std:", transformed.std())

#     # Hàm align dùng dlib (chỉ dùng nếu backbone=='edgeface' và dataset chưa pre-aligned)
#     def align_face(img):  # img là PIL Image
#         try:
#             img_cv = np.array(img)[:,:,::-1]  # PIL RGB -> OpenCV BGR
#             faces = detector(img_cv, 1)
#             if not faces:
#                 return TF.to_tensor(img_cv[:,:,::-1])  # Fallback if no face
#             main_face = max(faces, key=lambda f: f.width() * f.height())
#             x, y, w, h = main_face.left(), main_face.top(), main_face.width(), main_face.height()
#             margin = int(w * 0.35)
#             x1, y1 = max(0, x - margin), max(0, y - margin)
#             x2, y2 = min(img_cv.shape[1], x + w + margin), min(img_cv.shape[0], y + h + margin)
#             crop = img_cv[y1:y2, x1:x2]
#             return TF.to_tensor(crop[:,:,::-1])  # BGR -> RGB tensor
#         except Exception as e:
#             print(f"Align error: {e}")
#             return TF.to_tensor(img)  # Fallback

#     # Align only for EdgeFace if dataset not pre-aligned (FaceScrub is pre-aligned 112x112)
#     align_transform = nn.Identity()  # Skip align since dataset is pre-aligned
#     # Uncomment below if you want to enable alignment for EdgeFace
#     # align_transform = transforms.Lambda(align_face) if backbone == 'edgeface' else nn.Identity()

#     # Normalize and resize conditional
#     # if backbone == 'edgeface':
#     #     # norm_mean = [0.618, 0.465, 0.393]
#     #     # norm_std = [0.238, 0.202, 0.190]
#     #     norm_mean = (0.5, 0.5, 0.5) #Norm [-1 1] thay vì. [0 1]
#     #     norm_std = (0.5, 0.5, 0.5)
#     #     resize_crop_size = 120
#     #     crop_size = 112
        
#     # Normalize and resize conditional
#     if backbone == 'edgeface':
#         # TỰ ĐỘNG PHÁT HIỆN DATA 32x32 QUA ĐƯỜNG DẪN
#         if '32x32' in data_dir.lower() or '32' in data_dir.lower():
#             # DÙNG CHO DATASET 32x32 → CHỈ ĐỔI KÍCH THƯỚC
#             norm_mean = (0.5, 0.5, 0.5)  # GIỮ NGUYÊN [-1, 1]
#             norm_std  = (0.5, 0.5, 0.5)
#             resize_crop_size = 35
#             crop_size = 32
#             print("EdgeFace: Phát hiện dataset 32x32 → Resize(35) + Crop(32)")
#         else:
#             # MẶC ĐỊNH 112x112
#             norm_mean = (0.5, 0.5, 0.5)
#             norm_std  = (0.5, 0.5, 0.5)
#             resize_crop_size = 120
#             crop_size = 112
#             print("EdgeFace: Dùng Resize(120) + Crop(112)")

#     else:  # resnet (gốc OPQN) ← ĐÚNG VỊ TRÍ
#         if dataset == "vggface2" or cross_eval:
#             norm_mean = (0.5, 0.5, 0.5)
#             norm_std  = (0.5, 0.5, 0.5)
#             resize_crop_size = 120
#             crop_size = 112
#         else:  # facescrub
#             norm_mean = [0.639, 0.479, 0.404]
#             norm_std  = [0.216, 0.183, 0.171]
#             resize_crop_size = 35
#             crop_size = 32

#     # Transforms
#     if cross_eval:
#         transform_train = nn.Sequential(
#             align_transform,
#             transforms.Resize(resize_crop_size),
#             transforms.CenterCrop(crop_size),
#             transforms.ConvertImageDtype(torch.float),
#             transforms.Normalize(norm_mean, norm_std),
#         )
#         transform_test = transform_train
#     else:
#         if dataset == "vggface2":
#             transform_train = nn.Sequential(
#                 align_transform,
#                 transforms.Resize(resize_crop_size),
#                 transforms.RandomCrop(crop_size),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomGrayscale(p=0.2),  # Thêm giống EdgeFace
#                 transforms.GaussianBlur(kernel_size=3),  # Thêm giống EdgeFace
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize(norm_mean, norm_std),
#             )
#             transform_test = nn.Sequential(
#                 align_transform,
#                 transforms.Resize(resize_crop_size),
#                 transforms.CenterCrop(crop_size),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize(norm_mean, norm_std),
#             )
#         else:
#             transform_train = nn.Sequential(
#                 align_transform,
#                 transforms.Resize(resize_crop_size),
#                 transforms.RandomCrop(crop_size),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomGrayscale(p=0.2),  # Thêm giống EdgeFace
#                 transforms.GaussianBlur(kernel_size=3),  # Thêm giống EdgeFace
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize(norm_mean, norm_std),
#             )
#             transform_test = nn.Sequential(
#                 align_transform,
#                 transforms.Resize(resize_crop_size),
#                 transforms.CenterCrop(crop_size),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize(norm_mean, norm_std),
#             )

#     return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}








# import torch
# import torchvision.transforms as transforms
# from torchvision import datasets
# import os

# def get_datasets_transform(dataset, data_dir="/kaggle/input/facescrub-0210-3", cross_eval=False):
#     to_tensor = transforms.ToTensor()

#     # Define paths for FaceScrub processed data
#     if dataset == "facescrub":
#         train_path = os.path.join(data_dir, "facescrub", "train", "actors")
#         test_path = os.path.join(data_dir, "facescrub", "test", "actors")
#     elif dataset == "vggface2":
#         if cross_eval:
#             train_path = os.path.join(data_dir, "vggface2", "cross_train")
#             test_path = os.path.join(data_dir, "vggface2", "cross_test")
#         else:
#             train_path = os.path.join(data_dir, "vggface2", "train")
#             test_path = os.path.join(data_dir, "vggface2", "test")
#     else:
#         train_path = os.path.join(data_dir, dataset, "train")
#         test_path = os.path.join(data_dir, dataset, "test")

#     # Load datasets with debug print
#     trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
#     testset = datasets.ImageFolder(root=test_path, transform=to_tensor)
#     print(f"Train path: {train_path}")  # Debug
#     print(f"Test path: {test_path}")    # Debug
#     #print(f"Train classes: {trainset.classes}")  # Debug: Print number of identities
#     #print(f"Test classes: {testset.classes}")    # Debug: Print number of identities

#     if cross_eval:
#         transform_train = torch.nn.Sequential(
#             transforms.Resize(120),
#             transforms.CenterCrop(112),
#             transforms.ConvertImageDtype(torch.float),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         )
#         transform_test = transform_train
#     else:
#         if dataset == "vggface2":
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.RandomCrop(112),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.CenterCrop(112),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#         else:
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.RandomCrop(32),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.CenterCrop(32),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )

#     return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}






# import torch
# import torchvision.transforms as transforms
# from torchvision import datasets
# import os
# from torch.utils.data import ConcatDataset

# def get_datasets_transform(dataset, data_dir="/kaggle/input/facescrub-0210-2", cross_eval=False):
#     to_tensor = transforms.ToTensor()

#     # Define paths for FaceScrub processed data
#     if dataset == "facescrub":
#         # Kết hợp cả actor và actress cho train và test
#         train_paths = [
#             os.path.join(data_dir, "facescrub", "train", "actor"),
#             os.path.join(data_dir, "facescrub", "train", "actress")
#         ]
#         test_paths = [
#             os.path.join(data_dir, "facescrub", "test", "actor"),
#             os.path.join(data_dir, "facescrub", "test", "actress")
#         ]
#     elif dataset == "vggface2":
#         if cross_eval:
#             train_path = os.path.join(data_dir, "vggface2", "cross_train")
#             test_path = os.path.join(data_dir, "vggface2", "cross_test")
#         else:
#             train_path = os.path.join(data_dir, "vggface2", "train")
#             test_path = os.path.join(data_dir, "vggface2", "test")
#     else:
#         train_path = os.path.join(data_dir, dataset, "train")
#         test_path = os.path.join(data_dir, dataset, "test")

#     # Load datasets with debug print
#     if dataset == "facescrub":
#         trainsets = [datasets.ImageFolder(root=path, transform=to_tensor) for path in train_paths]
#         testsets = [datasets.ImageFolder(root=path, transform=to_tensor) for path in test_paths]
#         trainset = ConcatDataset(trainsets)
#         testset = ConcatDataset(testsets)
#     else:
#         trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
#         testset = datasets.ImageFolder(root=test_path, transform=to_tensor)

#     print(f"Train paths: {train_paths}")  # Debug
#     print(f"Test paths: {test_paths}")    # Debug
#     print(f"Train classes: {trainset.classes}")  # Debug: Print number of identities
#     print(f"Test classes: {testset.classes}")    # Debug: Print number of identities

#     if cross_eval:
#         transform_train = torch.nn.Sequential(
#             transforms.Resize(120),
#             transforms.CenterCrop(112),
#             transforms.ConvertImageDtype(torch.float),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         )
#         transform_test = transform_train
#     else:
#         if dataset == "vggface2":
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.RandomCrop(112),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.CenterCrop(112),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#         else:
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.RandomCrop(32),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.CenterCrop(32),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )

#     # Áp dụng transform sau khi ghép dataset
#     trainset.transform = transform_train
#     testset.transform = transform_test

#     return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}







# import torch
# import torchvision.transforms as transforms
# from torchvision import datasets

# import os

# def get_datasets_transform(dataset, data_dir="/kaggle/input/facescrub-0210-1", cross_eval=False):
#     to_tensor = transforms.ToTensor()
#     if dataset!="vggface2":
#         trainPaths = os.path.join(data_dir, dataset, "train") 
#         testPaths = os.path.join(data_dir, dataset, "test")
#     else:
#         if cross_eval: # vgggface2 cross-dataset retrieval uses another train-test splits from standard retrieval
#             trainPaths = os.path.join(data_dir, "vggface2", "cross_train") 
#             testPaths = os.path.join(data_dir, "vggface2", "cross_test")
#         else:
#             trainPaths = os.path.join(data_dir, "vggface2", "train") 
#             testPaths = os.path.join(data_dir, "vggface2", "test")
#     trainset = datasets.ImageFolder(root=trainPaths, transform=to_tensor)
#     testset = datasets.ImageFolder(root=testPaths, transform=to_tensor)
#     if cross_eval:
#         transform_train = torch.nn.Sequential(
#                     transforms.Resize(120),
#                     transforms.CenterCrop(112),
#                     # transforms.RandomHorizontalFlip(),
#                     transforms.ConvertImageDtype(torch.float),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         )

#         transform_test = transform_train

#     else:
#         if dataset=="vggface2":
#             transform_train = torch.nn.Sequential(
#                     transforms.Resize(120),
#                     transforms.RandomCrop(112),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ConvertImageDtype(torch.float),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )

#             transform_test = torch.nn.Sequential(
#                     transforms.Resize(120),
#                     transforms.CenterCrop(112),
#                     transforms.ConvertImageDtype(torch.float),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
        
#         else:
#             transform_train = torch.nn.Sequential(
#                     transforms.Resize(35), 
#                     transforms.RandomCrop(32),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ConvertImageDtype(torch.float),
#                     transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )

#             transform_test = torch.nn.Sequential(
#                     transforms.Resize(35), 
#                     transforms.CenterCrop(32),
#                     transforms.ConvertImageDtype(torch.float),
#                     transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )
#     return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}
    





# import torch
# import torchvision.transforms as transforms
# from torchvision import datasets
# import os

# def get_datasets_transform(dataset, data_dir="./data", cross_eval=False):
#     to_tensor = transforms.ToTensor()

#     # Auto detect Kaggle and use /kaggle/input/ processed paths
#     if 'kaggle' in os.environ.get('PWD', ''):
#         if dataset == 'facescrub':
#             base_path = '/kaggle/input/processed_facescrub/processed_facescrub/'  # Đường dẫn đúng với dataset của bạn
#         else:
#             base_path = data_dir  # Fallback cho dataset khác
#     else:
#         base_path = data_dir  # Cục bộ

#     # Define paths with folder existence check
#     train_path = os.path.join(base_path, "train")
#     test_path = os.path.join(base_path, "test")
#     if not os.path.exists(train_path):
#         train_path = base_path  # Fallback nếu không có split
#     if not os.path.exists(test_path):
#         test_path = train_path  # Fallback

#     # Debug print
#     print(f"Dataset: {dataset}, Cross-eval: {cross_eval}")
#     print(f"Train path: {train_path}, Test path: {test_path}")

#     trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
#     testset = datasets.ImageFolder(root=test_path, transform=to_tensor)

#     # Transforms
#     if cross_eval:
#         transform_train = torch.nn.Sequential(
#             transforms.Resize(120),
#             transforms.CenterCrop(112),
#             transforms.ConvertImageDtype(torch.float),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         )
#         transform_test = transform_train
#     else:
#         if dataset == "vggface2":
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.RandomCrop(112),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.CenterCrop(112),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#         else:
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.RandomCrop(32),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.CenterCrop(32),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )

#     return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}




# import torch
# import torchvision.transforms as transforms
# from torchvision import datasets
# import os

# def get_datasets_transform(dataset, data_dir="./data", cross_eval=False):
#     to_tensor = transforms.ToTensor()

#     # Auto detect Kaggle and use /kaggle/input/ processed paths
#     if 'kaggle' in os.environ.get('PWD', ''):
#         if dataset == 'facescrub':
#             base_path = '/kaggle/input/processed-facescrub/processed-facescrub/'  # Thay bằng tên dataset bạn upload
#         else:
#             base_path = data_dir  # Fallback cho dataset khác
#     else:
#         base_path = data_dir  # Cục bộ

#     # Define paths with folder existence check
#     if dataset != "vggface2":
#         train_path = os.path.join(base_path, "train")  # Thay vì dataset/train
#         test_path = os.path.join(base_path, "test")    # Thay vì dataset/test
#         if not os.path.exists(train_path):
#             train_path = os.path.join(base_path)  # Fallback nếu không có split
#         if not os.path.exists(test_path):
#             test_path = train_path  # Fallback
#     else:
#         if cross_eval:  # vggface2 cross-dataset
#             train_path = os.path.join(base_path, "cross_train") if os.path.exists(os.path.join(base_path, "cross_train")) else os.path.join(base_path, "train")
#             test_path = os.path.join(base_path, "cross_test") if os.path.exists(os.path.join(base_path, "cross_test")) else os.path.join(base_path, "test")
#         else:
#             train_path = os.path.join(base_path, "train")
#             test_path = os.path.join(base_path, "test")

#     # Debug print
#     print(f"Dataset: {dataset}, Cross-eval: {cross_eval}")
#     print(f"Train path: {train_path}, Test path: {test_path}")

#     trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
#     testset = datasets.ImageFolder(root=test_path, transform=to_tensor)

#     # Transforms (giữ nguyên)
#     if cross_eval:
#         transform_train = torch.nn.Sequential(
#             transforms.Resize(120),
#             transforms.CenterCrop(112),
#             transforms.ConvertImageDtype(torch.float),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         )
#         transform_test = transform_train
#     else:
#         if dataset == "vggface2":
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.RandomCrop(112),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(120),
#                 transforms.CenterCrop(112),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             )
#         else:
#             transform_train = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.RandomCrop(32),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )
#             transform_test = torch.nn.Sequential(
#                 transforms.Resize(35),
#                 transforms.CenterCrop(32),
#                 transforms.ConvertImageDtype(torch.float),
#                 transforms.Normalize([0.639, 0.479, 0.404], [0.216, 0.183, 0.171])
#             )

#     return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}