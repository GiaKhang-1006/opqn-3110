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
from torch.optim.lr_scheduler import LambdaLR
from utils import Logger, AverageMeter, compute_quant, compute_quant_indexing, PqDistRet_Ortho, PqDistRet_Ortho_safe
from backbone import resnet20_pq, SphereNet20_pq, EdgeFaceBackbone
from margin_metric import OrthoPQ, CosFace
from data_loader import get_datasets_transform
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR

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
parser.add_argument('--pretrain_cosface', action='store_true', help='Pretrain with CosFace loss before OrthoPQ')
parser.add_argument('--s_cosface', default=30.0, type=float, help='scale s for CosFace')
parser.add_argument('--m_cosface', default=0.2, type=float, help='margin m for CosFace')
parser.add_argument('--max_norm', default=0.5, type=float, help='gradient clipping max norm for pre-train')
parser.add_argument('--epochs_cosface', default=50, type=int, help='number of epochs for CosFace pre-training')
parser.add_argument('--lr_backbone', default=0.0001, type=float, help='learning rate for backbone in pre-train CosFace')
parser.add_argument('--input_size', type=int, default=112, help='Input size for model: 32 or 112')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--scheduler_type', default='step', choices=['step', 'plateau'], help='Scheduler type: step (paper) or plateau')
parser.add_argument('--freeze', action='store_true', help='Freeze backbone')

try:
    args = parser.parse_args()
except Exception as e:
    print(f"Parser error: {e}")
    sys.exit(1)

trainset, testset = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['dataset']
transform_train, transform_test = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['transform']

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

    if args.cross_dataset or args.dataset == "vggface2":
        if args.backbone == 'edgeface':
            net = EdgeFaceBackbone(feature_dim=feature_dim)
        else:
            net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
    else:
        if args.backbone == 'edgeface':
            net = EdgeFaceBackbone(feature_dim=feature_dim)
        else:
            net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)

    net = nn.DataParallel(net).to(device)
    cudnn.benchmark = True

    if args.pretrain_cosface:
        print("Pre-training with CosFace loss...")
        feature_dim = 516 if args.len and args.len[0] == 36 else 512
        print(f"Selected feature_dim: {feature_dim} for code length: {args.len[0] if args.len else 'N/A'}")
        
        metric = CosFace(in_features=feature_dim, out_features=num_classes, s=args.s_cosface, m=args.m_cosface)
        net = EdgeFaceBackbone(feature_dim=feature_dim)
        metric = nn.DataParallel(metric).to(device)
        net = nn.DataParallel(net).to(device)
        cudnn.benchmark = True
        
        criterion = nn.CrossEntropyLoss()
        for name, param in net.named_parameters():
            if 'conv1' in name or 'layer1' in name:
                param.requires_grad = False
        optimizer = optim.AdamW([
            {'params': [p for p in net.parameters() if p.requires_grad], 'lr': args.lr_backbone},
            {'params': metric.parameters(), 'lr': args.lr_backbone * 10}
        ], weight_decay=args.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_cosface)
        checkpoint_dir = '/kaggle/working/opqn-3110/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(args.epochs_cosface):
            net.train()
            metric.train()
            losses = AverageMeter()
            grad_norm_backbone = 0
            grad_norm_metric = 0
            correct = 0
            total = 0
            start = time.time()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                transformed_images = transform_train(inputs)
                features = net(transformed_images)
                outputs = metric(features, targets)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                grad_norm_b = torch.norm(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])).item()
                grad_norm_m = torch.norm(torch.cat([p.grad.flatten() for p in metric.parameters() if p.grad is not None])).item()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
                torch.nn.utils.clip_grad_norm_(metric.parameters(), max_norm=args.max_norm)
                optimizer.step()
                losses.update(loss.item(), len(inputs))
                grad_norm_backbone += grad_norm_b
                grad_norm_metric += grad_norm_m
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            avg_loss = losses.avg
            avg_grad_norm_backbone = grad_norm_backbone / len(train_loader)
            avg_grad_norm_metric = grad_norm_metric / len(train_loader)
            accuracy = 100. * correct / total
            epoch_elapsed = time.time() - start
            print(f"Pre-train Epoch {epoch+1} | Loss: {avg_loss:.4f} | Grad_norm_backbone: {avg_grad_norm_backbone:.4f} | Grad_norm_metric: {avg_grad_norm_metric:.4f} | Accuracy: {accuracy:.2f}%")

            if (epoch + 1) % 5 == 0:
                net.eval()
                metric.eval()
                test_correct = 0
                test_total = 0
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        features = net(transform_test(inputs))
                        outputs = metric(features, targets)
                        _, predicted = outputs.max(1)
                        test_total += targets.size(0)
                        test_correct += predicted.eq(targets).sum().item()
                test_accuracy = 100. * test_correct / test_total
                print(f"[Test Phase] Epoch: {epoch+1} | Test Accuracy: {test_accuracy:.2f}%")
            scheduler.step()

        print("Saving pre-trained model...")
        torch.save({'backbone': net.state_dict()}, os.path.join(checkpoint_dir, save_path))
        return

    if args.load:
        load_path = args.load[0]
        if os.path.isabs(load_path):
            checkpoint_path = load_path
        else:
            checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
            checkpoint_path = os.path.join(checkpoint_dir, load_path)

        if not os.path.exists(checkpoint_path):
            print(f"Lỗi: Không tìm thấy file checkpoint tại: {checkpoint_path}")
            sys.exit(1)
        
        print(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['backbone'])

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
    code_books /= torch.norm(code_books, dim=1, keepdim=True)
    print("Codebook norms:", [torch.norm(code_books[i], dim=1).mean().item() for i in range(num)])

    metric = OrthoPQ(in_features=feature_dim, out_features=num_classes, num_books=num, num_words=words, code_books=code_books, sc=args.sc, m=args.margin)
    metric = nn.DataParallel(metric).to(device)

    criterion = nn.CrossEntropyLoss()
    num_books = num
    num_words = words
    len_word = int(feature_dim / num_books)
    len_bit = int(num_books * math.log(num_words, 2))
    assert length == len_bit, f"Code length mismatch: expected {length}-bit, got {len_bit}-bit"
    print("num. of codebooks: ", num_books)
    print("num. of words per book: ", num_words)
    print("dim. of word: ", len_word)
    print("code length: %d-bit \t learning rate: %.3f \t scale length: %d \t penalty margin: %.2f \t balance_weight: %.3f" % 
          (len_bit, args.lr, metric.module.s, metric.module.m, args.miu))

    optimizer_params = [
        {'params': metric.parameters(), 'lr': args.lr},
        {'params': [p for p in net.parameters() if p.requires_grad], 'lr': args.lr * 0.1}
    ]
    optimizer = optim.SGD(optimizer_params, weight_decay=args.wd, momentum=0.9)

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

    EPOCHS = 300 if args.dataset in ["facescrub", "cfw", "youtube"] else 160
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
            inputs, targets = inputs.to(device), targets.to(device)
            transformed_images = transform_train(inputs)
            features = net(transformed_images)
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
                best_mAP = mAP
                print('Saving..')
                checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({'backbone': net.state_dict(), 'mlp': metric.module.mlp}, os.path.join(checkpoint_dir, save_path))
                best_epoch = epoch + 1
    time_elapsed = time.time() - since
    print("Training Completed in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best mAP {:.4f} at epoch {}".format(best_mAP, best_epoch))
    print("Model saved as %s" % save_path)

def test(load_path, length, num, words, feature_dim=512):
    print("===============evaluation on model %s===============" % load_path)
    num_classes = len(trainset.classes)
    num_classes_test = len(testset.classes)
    print("number of train identities: ", num_classes)
    print("number of test identities: ", num_classes_test)
    print("number of training images: ", len(trainset))
    print("number of test images: ", len(testset))
    print("number of training batches per epoch:", len(train_loader))
    print("number of testing batches per epoch:", len(test_loader))

    if args.cross_dataset:
        if args.backbone == 'edgeface':
            net = EdgeFaceBackbone(feature_dim=feature_dim)
        else:
            net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
    else:
        if args.dataset in ["facescrub", "cfw", "youtube"]:
            if args.backbone == 'edgeface':
                net = EdgeFaceBackbone(feature_dim=feature_dim)
            else:
                net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
        else:
            if args.backbone == 'edgeface':
                net = EdgeFaceBackbone(feature_dim=feature_dim)
            else:
                net = resnet20_pq(num_layers=20, feature_dim=feature_dim)

    net = nn.DataParallel(net).to(device)

    checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
    checkpoint_path = os.path.join(checkpoint_dir, load_path)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['backbone'])
    mlp_weight = checkpoint['mlp']
    len_word = int(feature_dim / num)
    net.eval()
    with torch.no_grad():
        index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
        start = datetime.now()
        query_features, test_labels = compute_quant(transform_test, test_loader, net, device)
        if args.dataset != "vggface2":
            mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=5)
        else:
            mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=10)

        time_elapsed = datetime.now() - start
        print("Query completed in %d ms" % int(time_elapsed.total_seconds() * 1000))
        print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

if __name__ == "__main__":
    save_dir = 'log'
    if args.evaluate:
        if len(args.load) != len(args.num) or len(args.load) != len(args.len) or len(args.load) != len(args.words):
            print("Warning: Args lengths don't match. Adjusting to shortest length.")
            min_len = min(len(args.load), len(args.num), len(args.len), len(args.words))
            args.load = args.load[:min_len]
            args.num = args.num[:min_len]
            args.len = args.len[:min_len]
            args.words = args.words[:min_len]
        for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
            if args.cross_dataset:
                feature_dim = num_s * words_s
            else:
                if args.dataset != "vggface2":
                    if args.len[i] != 36:
                        feature_dim = 512
                    else:
                        feature_dim = 516
                else:
                    feature_dim = num_s * words_s
            test(args.load[i], args.len[i], num_s, words_s, feature_dim=feature_dim)
    else:
        if args.pretrain_cosface:
            pass  # Không cần num, words, len → bỏ kiểm tra
        else:
            if len(args.save) != len(args.num) or len(args.save) != len(args.len) or len(args.save) != len(args.words):
                print("Warning: Args lengths don't match. Adjusting to shortest length.")
                min_len = min(len(args.save), len(args.num), len(args.len), len(args.words))
                args.save = args.save[:min_len]
                args.num = args.num[:min_len]
                args.len = args.len[:min_len]
                args.words = args.words[:min_len]
            for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
                sys.stdout = Logger(os.path.join(save_dir,
                    str(args.len[i]) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
                print("[Configuration] Training on dataset: %s\n  Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n num_books: %d\n num_words: %d"
                    % (args.dataset, args.len[i], args.bs, args.lr, num_s, words_s))
                print("HyperParams:\nmargin: %.3f\t miu: %.4f" % (args.margin, args.miu))
                if args.dataset != "vggface2":
                    if args.len[i] != 36:
                        feature_dim = 512
                    else:
                        feature_dim = 516
                    train(args.save[i], args.len[i], num_s, words_s, feature_dim=feature_dim)
                else:
                    feature_dim = num_s * words_s
                    train(args.save[i], args.len[i], num_s, words_s, feature_dim=feature_dim)