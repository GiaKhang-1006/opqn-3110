import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from datetime import datetime
import torch.distributions as Distributions
import math
import argparse
import sys
import time
import os
from utils import Logger, AverageMeter, compute_quant, compute_quant_indexing, PqDistRet_Ortho, PqDistRet_Ortho_safe
from backbone import resnet20_pq, SphereNet20_pq, EdgeFaceBackbone
from margin_metric import OrthoPQ
from data_loader import get_datasets_transform
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser(description='PyTorch Implementation of Orthonormal Product Quantization for Scalable Face Image Retrieval')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate mode turned on')
parser.add_argument('-c', '--cross-dataset', action='store_true', help='generalize on unseen identities')
parser.add_argument('--bs', type=int, default=256, help='Batch size of each iteration')
parser.add_argument('--save', nargs='+', help='path to saving models, accept multiple arguments as list')
parser.add_argument('--load', nargs='+', help='path to loading models, accept multiple arguments as list')
parser.add_argument('--len', nargs='+', type=int, help='length of hashing codes, accept multiple arguments as list')
parser.add_argument('--dataset', type=str, default='facescrub', help='which dataset for training (one of facescrub, youtube, CFW, and VGGFace2)')
parser.add_argument('--num', nargs='+', type=int, help='num. of codebooks, could be 4, 8...')
parser.add_argument('--words', nargs='+', type=int, default=[256, 256, 256, 256], help='num of words, should be exponential of 2')
parser.add_argument('--margin', default=0.4, type=float, help='margin of cosine similarity')
parser.add_argument('--miu', default=0.1, type=float, help='Balance weight of redundancy loss')
parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'edgeface'], help='Backbone type: resnet or edgeface')
parser.add_argument('--data_dir', type=str, default='/kaggle/input/facescrub-0210-3', help='Data direction on kaggle for multiple dataset')
parser.add_argument('--sc', default=30, type=float, help='scale s for initialize metric')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--scheduler_type', default='step', choices=['step', 'plateau'], help='Scheduler type: step (paper) or plateau')
parser.add_argument('--freeze', action='store_true', help='Freeze backbone')

try:
    args = parser.parse_args()
except Exception as e:
    print(f"Parser error: {e}")
    sys.exit(1)

# trainset, testset = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['dataset']
# transform_train, transform_test = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['transform']
# THAY THẾ TOÀN BỘ KHỐI NÀY
data_dict = get_datasets_transform(
    args.dataset, 
    args.data_dir, 
    cross_eval=args.cross_dataset, 
    backbone=args.backbone
)
trainset, testset = data_dict['dataset']
transform_train, transform_test = data_dict['transform']

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=4)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.cuda.manual_seed_all(1)

def train(save_path, length, num, words, feature_dim):
    best_acc = 0
    best_mAP = 0
    best_epoch = 1
    print('==> Building model..')
    num_classes = len(trainset.classes)
    print("number of identities: ", num_classes)
    print("number of training images: ", len(trainset))
    print("number of test images: ", len(testset))
    print("number of training batches per epoch:", len(train_loader))
    print("number of testing batches per epoch:", len(test_loader))
    
    # KIỂM TRA OVERLAP IDENTITY (SEEN RETRIEVAL)
    train_ids = set(trainset.targets)
    test_ids = set(testset.targets)
    overlap = train_ids.intersection(test_ids)
    print(f"Overlap identities (seen): {len(overlap)} / {len(test_ids)}")
    assert len(overlap) == len(test_ids), "LỖI: Test identities KHÔNG có trong train! (Phải là seen retrieval)"
    # -----------------------------------

    d = int(feature_dim / num)
    matrix = torch.randn(d, d)
    for k in range(d):
        for j in range(d):
            matrix[j, k] = math.cos((j+0.5)*k*math.pi/d)
    matrix[:, 0] /= math.sqrt(2)
    matrix /= math.sqrt(d/2)
    code_books = torch.Tensor(num, d, words)
    code_books[0] = matrix[:, :words]
    for i in range(1, num):
        code_books[i] = matrix @ code_books[i-1]

    if args.cross_dataset or args.dataset == "vggface2":
        if args.backbone == 'edgeface':
            net = EdgeFaceBackbone(feature_dim=feature_dim)
        else:
            net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
        metric = OrthoPQ(in_features=feature_dim, out_features=num_classes, num_books=num, code_books=code_books, num_words=words, sc=args.sc, m=args.margin)
    else:
        if args.backbone == 'edgeface':
            net = EdgeFaceBackbone(feature_dim=feature_dim)
        else:
            net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
        metric = OrthoPQ(in_features=feature_dim, out_features=num_classes, num_books=num, code_books=code_books, num_words=words, sc=args.sc, m=args.margin)

    num_books = metric.num_books
    len_word = metric.len_word
    num_words = metric.num_words
    len_bit = int(num_books * math.log(num_words, 2))
    assert length == len_bit, "something went wrong with code length"
    criterion = nn.CrossEntropyLoss()
    print("num. of codebooks: ", num_books)
    print("num. of words per book: ", num_words)
    print("dim. of word: ", len_word)
    print("code length: %d-bit \t learning rate: %.3f \t scale length: %d \t penalty margin: %.2f \t balance_weight: %.3f" % 
          (len_bit, args.lr, metric.s, metric.m, args.miu))
    net = nn.DataParallel(net).to(device)
    if args.backbone == 'edgeface' and args.freeze:
        print("Freezing EdgeFace Backbone parameters.")
        for param in net.module.backbone.parameters():
            param.requires_grad = False
    metric = nn.DataParallel(metric).to(device)
    cudnn.benchmark = True

    if args.dataset in ["facescrub", "cfw", "youtube"]:
        optimizer_params = [{'params': metric.parameters(), 'lr': args.lr}]
        if any(p.requires_grad for p in net.parameters()):
            optimizer_params.append({'params': [p for p in net.parameters() if p.requires_grad], 'lr': args.lr})
        optimizer = optim.SGD(optimizer_params, weight_decay=args.wd, momentum=0.9)
        EPOCHS = 200
    else:
        optimizer_params = [{'params': metric.parameters(), 'lr': args.lr}]
        if any(p.requires_grad for p in net.parameters()):
            optimizer_params.append({'params': [p for p in net.parameters() if p.requires_grad], 'lr': args.lr})
        optimizer = optim.SGD(optimizer_params, weight_decay=args.wd, momentum=0.9)
        EPOCHS = 160

    if args.scheduler_type == 'step':
        class adjust_lr:
            def __init__(self, step=35, decay=0.5):
                self.step = step
                self.decay = decay
            def adjust(self, optimizer, epoch):
                lr = args.lr * (self.decay ** (epoch // self.step))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                return lr
        scheduler = adjust_lr()
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    since = time.time()
    best_loss = 1e3

    for epoch in range(EPOCHS):
        print('==> Epoch: %d' % (epoch+1))
        net.train()
        losses = AverageMeter()
        if args.scheduler_type == 'step':
            scheduler.adjust(optimizer, epoch)
        start = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            #transformed_images = transform_train(inputs)
            #features = net(transformed_images)
            features = net(inputs)
            output1, output2, xc_probs = metric(features, targets)
            loss_clf1 = [criterion(output1[:, i, :], targets) for i in range(num_books)]
            loss_clf2 = [criterion(output2[:, i, :], targets) for i in range(num_books)]
            loss_clf = 0.5 * (sum(loss_clf1) / len(loss_clf1) + sum(loss_clf2) / len(loss_clf2))
            xc_entropy = [Distributions.categorical.Categorical(probs=xc_probs[:, i, :]).entropy().sum() for i in range(num_books)]
            loss_entropy = sum(xc_entropy) / (num_books * len(inputs))
            loss = loss_clf + args.miu * loss_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), len(inputs))

        epoch_elapsed = time.time() - start
        print('Epoch %d | Loss: %.4f' % (epoch+1, losses.avg))
        print("Epoch Completed in {:.0f}min {:.0f}s".format(epoch_elapsed // 60, epoch_elapsed % 60))
        if args.scheduler_type != 'step':
            scheduler.step(losses.avg)

        if (epoch+1) % 5 == 0:
            net.eval()
            with torch.no_grad():
                mlp_weight = metric.module.mlp
                index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
                queries, test_labels = compute_quant(transform_test, test_loader, net, device)
                start = time.time()
                mAP, top_k = PqDistRet_Ortho(queries, test_labels, train_labels, index, mlp_weight, len_word, num_books, device, top=50)
                time_elapsed = time.time() - start
                print("Code generated in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
                print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

            if losses.avg < best_loss:
                best_loss = losses.avg
                print('Saving..')
                checkpoint_dir = '/kaggle/working/opqn-3110/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({'backbone': net.state_dict(), 'mlp': metric.module.mlp}, os.path.join(checkpoint_dir, save_path))
                best_epoch = epoch + 1
    time_elapsed = time.time() - since
    print("Training Completed in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best mAP {:.4f} at epoch {}".format(best_mAP, best_epoch))
    print("Model saved as %s" % save_path)

if __name__ == "__main__":
    save_dir = 'log'
    if args.evaluate:
        assert len(args.load) == len(args.num), 'model paths must be in line with # code lengths'
        for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
            if args.cross_dataset:
                feature_dim = num_s * words_s
            else:
                if args.dataset!="vggface2":
                    if args.len[i] != 36:
                        feature_dim = 512
                    else:
                        feature_dim = 516
                else:
                    feature_dim=num_s * words_s
            # Gọi test nếu có, nhưng code gốc không có, thêm nếu cần
    else:
        assert len(args.save) == len(args.num) and len(args.save) == len(args.words), 'model paths must be in line with # code lengths'
        for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
            sys.stdout = Logger(os.path.join(save_dir,
                str(args.len[i]) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
            print("[Configuration] Training on dataset: %s\n  Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n num_books: %d\n num_words: %d"
            %(args.dataset, args.len[i], args.bs, args.lr, num_s, words_s))
            print("HyperParams:\nmargin: %.3f\t miu: %.4f" % (args.margin, args.miu))
            if args.dataset!="vggface2":
                if args.len[i] != 36:
                    feature_dim = 512
                else:
                    feature_dim = 516
            else:
                feature_dim=num_s * words_s
          
            train(args.save[i], args.len[i], num_s, words_s, feature_dim=feature_dim)