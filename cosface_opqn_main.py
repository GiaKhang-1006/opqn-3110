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
        # Chọn feature_dim dựa trên args.len
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
        ], weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_cosface)
        checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
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

    # <<< THAY ĐỔI BẮT ĐẦU >>>
    if args.load:
        load_path = args.load[0]
        # Kiểm tra nếu là đường dẫn tuyệt đối (ví dụ: /kaggle/input/...)
        if os.path.isabs(load_path):
            checkpoint_path = load_path
        # Nếu không, dùng đường dẫn tương đối trong thư mục checkpoint
        else:
            checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
            checkpoint_path = os.path.join(checkpoint_dir, load_path)

        # Kiểm tra xem file có tồn tại không trước khi load
        if not os.path.exists(checkpoint_path):
            print(f"Lỗi: Không tìm thấy file checkpoint tại: {checkpoint_path}")
            sys.exit(1)
        
        print(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['backbone'])
    # <<< THAY ĐỔI KẾT THÚC >>>

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
    optimizer = optim.SGD(optimizer_params, weight_decay=1e-3, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    EPOCHS = 300 if args.dataset in ["facescrub", "cfw", "youtube"] else 160
    since = time.time()
    best_loss = 1e3

    for epoch in range(EPOCHS):
        print('==> Epoch: %d' % (epoch+1))
        net.train()
        metric.train()
        losses = AverageMeter()
        loss_clf_avg = AverageMeter()
        loss_entropy_avg = AverageMeter()
        grad_norm_backbone = 0
        grad_norm_metric = 0
        correct = 0
        total = 0
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
            grad_norm_b = torch.norm(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])).item()
            grad_norm_m = torch.norm(torch.cat([p.grad.flatten() for p in metric.parameters() if p.grad is not None])).item()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            torch.nn.utils.clip_grad_norm_(metric.parameters(), max_norm=args.max_norm)
            optimizer.step()
            losses.update(loss.item(), len(inputs))
            loss_clf_avg.update(loss_clf.item(), len(inputs))
            loss_entropy_avg.update(loss_entropy.item(), len(inputs))
            grad_norm_backbone += grad_norm_b
            grad_norm_metric += grad_norm_m
            _, predicted = output1[:, 0, :].max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = losses.avg
        avg_loss_clf = loss_clf_avg.avg
        avg_loss_entropy = loss_entropy_avg.avg
        avg_grad_norm_backbone = grad_norm_backbone / len(train_loader)
        avg_grad_norm_metric = grad_norm_metric / len(train_loader)
        accuracy = 100. * correct / total
        epoch_elapsed = time.time() - start
        print(f'Epoch: {epoch+1} | Loss_clf: {avg_loss_clf:.4f} | Loss_entropy: {avg_loss_entropy:.4f} | Total Loss: {avg_loss:.4f} | Grad_norm_backbone: {avg_grad_norm_backbone:.4f} | Grad_norm_metric: {avg_grad_norm_metric:.4f} | Accuracy: {accuracy:.2f}%')
        print("Epoch Completed in {:.0f}min {:.0f}s".format(epoch_elapsed // 60, epoch_elapsed % 60))
        scheduler.step(avg_loss)

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

            if avg_loss < best_loss:
                best_loss = avg_loss
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
    len_bit = int(num * math.log(words, 2))
    assert length == len_bit, "something went wrong with code length"

    print(f"=============== Evaluation on model {load_path} ===============")
    num_classes = len(trainset.classes)
    num_classes_test = len(testset.classes)
    print(f"Number of train identities: {num_classes}")
    print(f"Number of test identities: {num_classes_test}")
    print(f"Number of training images: {len(trainset)}")
    print(f"Number of test images: {len(testset)}")
    print(f"Number of training batches per epoch: {len(train_loader)}")
    print(f"Number of testing batches per epoch: {len(test_loader)}")

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

    # Kiểm tra nếu là đường dẫn tuyệt đối (ví dụ: /kaggle/input/...)
    if os.path.isabs(load_path):
        checkpoint_path = load_path
    else:
        checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
        checkpoint_path = os.path.join(checkpoint_dir, load_path)

    # Kiểm tra xem file có tồn tại không trước khi load
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} not found")
        sys.exit(1)
        
    print(f"Loading pretrained weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['backbone'])
    mlp_weight = checkpoint.get('mlp', None)  # Sử dụng get để tránh lỗi nếu 'mlp' không tồn tại

    len_word = int(feature_dim / num)
    net.eval()
    
    with torch.no_grad():
        # Tính index cho tập train
        index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
        
        # Đo thời gian truy vấn cho tập test
        start_total = time.perf_counter()  # Thời gian tổng
        query_features, test_labels = compute_quant(transform_test, test_loader, net, device)
        
        # Đo thời gian riêng cho tính mAP
        start_map = time.perf_counter()
        mAP, _ = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=len(trainset))
        map_time_ms = (time.perf_counter() - start_map) * 1000  # ms
        map_time_per_image = map_time_ms / len(testset)  # ms/image
        
        # In mAP và thời gian mAP
        print(f"[Evaluate Phase] mAP: {100. * float(mAP):.2f}%")
        print(f"mAP computation time: {map_time_ms:.2f} ms ({map_time_per_image:.4f} ms/image)")
        
        # Vòng lặp cho top-k từ 10 đến 100, step 10
        for k in range(10, 101, 10):
            _, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=k)
            print(f"[Evaluate Phase @ top-{k}] top_k: {100. * float(top_k):.2f}%")
        
        # Tổng thời gian (bao gồm compute_quant + mAP + top-k)
        total_query_time = (time.perf_counter() - start_total) * 1000  # ms
        avg_query_time = total_query_time / len(testset)  # ms/query
    
    print(f"Total query time (feature extraction + mAP + top-k): {total_query_time:.2f} ms")
    print(f"Average query time: {avg_query_time:.4f} ms/query")
    
if __name__ == "__main__":
    save_dir = 'log'
    if args.evaluate:
        if not args.load:
            print("Error: --load is required for evaluation mode")
            sys.exit(1)
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
        if not args.save:
            print("Error: --save is required for training mode")
            sys.exit(1)
        if args.pretrain_cosface:
            if not args.len:
                args.len = [36] # Mặc định 36 bits để trigger feature_dim=516
            sys.stdout = Logger(os.path.join(save_dir,
                'cosface_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
            print("[Configuration] Pre-training on dataset: %s\n Batch_size: %d\n learning rate backbone: %.6f\n learning rate metric: %.6f\n s: %.1f\n m: %.1f\n max_norm: %.1f\n epochs: %d" %
                  (args.dataset, args.bs, args.lr_backbone, args.lr_backbone * 10, args.s_cosface, args.m_cosface, args.max_norm, args.epochs_cosface))
            train(args.save[0], None, None, None, feature_dim=512)
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
                print("[Configuration] Training on dataset: %s\n Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n num_books: %d\n num_words: %d" %
                      (args.dataset, args.len[i], args.bs, args.lr, num_s, words_s))
                print("HyperParams:\nmargin: %.3f\t miu: %.4f" % (args.margin, args.miu))
                if args.dataset != "vggface2":
                    if args.len[i] != 36:
                        feature_dim = 512
                    else:
                        feature_dim = 516
                else:
                    feature_dim = num_s * words_s
                train(args.save[i], args.len[i], num_s, words_s, feature_dim=feature_dim)


# if args.pretrain_cosface:
    #     print("Pre-training with CosFace loss...")
    #     metric = CosFace(in_features=feature_dim, out_features=num_classes, s=args.s_cosface, m=args.m_cosface)
    #     metric = nn.DataParallel(metric).to(device)
    #     criterion = nn.CrossEntropyLoss()
    #     for name, param in net.named_parameters():
    #         if 'conv1' in name or 'layer1' in name:
    #             param.requires_grad = False
    #     optimizer = optim.AdamW([
    #         {'params': [p for p in net.parameters() if p.requires_grad], 'lr': args.lr_backbone},
    #         {'params': metric.parameters(), 'lr': args.lr_backbone * 10}
    #     ], weight_decay=5e-4)
    #     scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_cosface)
    #     checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
    #     os.makedirs(checkpoint_dir, exist_ok=True)

    #     for epoch in range(args.epochs_cosface):
    #         net.train()
    #         metric.train()
    #         losses = AverageMeter()
    #         grad_norm_backbone = 0
    #         grad_norm_metric = 0
    #         correct = 0
    #         total = 0
    #         start = time.time()
    #         for batch_idx, (inputs, targets) in enumerate(train_loader):
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             transformed_images = transform_train(inputs)
    #             features = net(transformed_images)
    #             outputs = metric(features, targets)
    #             loss = criterion(outputs, targets)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             grad_norm_b = torch.norm(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])).item()
    #             grad_norm_m = torch.norm(torch.cat([p.grad.flatten() for p in metric.parameters() if p.grad is not None])).item()
    #             torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
    #             torch.nn.utils.clip_grad_norm_(metric.parameters(), max_norm=args.max_norm)
    #             optimizer.step()
    #             losses.update(loss.item(), len(inputs))
    #             grad_norm_backbone += grad_norm_b
    #             grad_norm_metric += grad_norm_m
    #             _, predicted = outputs.max(1)
    #             total += targets.size(0)
    #             correct += predicted.eq(targets).sum().item()

    #         avg_loss = losses.avg
    #         avg_grad_norm_backbone = grad_norm_backbone / len(train_loader)
    #         avg_grad_norm_metric = grad_norm_metric / len(train_loader)
    #         accuracy = 100. * correct / total
    #         epoch_elapsed = time.time() - start
    #         print(f"Pre-train Epoch {epoch+1} | Loss: {avg_loss:.4f} | Grad_norm_backbone: {avg_grad_norm_backbone:.4f} | Grad_norm_metric: {avg_grad_norm_metric:.4f} | Accuracy: {accuracy:.2f}%")

    #         if (epoch + 1) % 5 == 0:
    #             net.eval()
    #             metric.eval()
    #             test_correct = 0
    #             test_total = 0
    #             with torch.no_grad():
    #                 for inputs, targets in test_loader:
    #                     inputs, targets = inputs.to(device), targets.to(device)
    #                     features = net(transform_test(inputs))
    #                     outputs = metric(features, targets)
    #                     _, predicted = outputs.max(1)
    #                     test_total += targets.size(0)
    #                     test_correct += predicted.eq(targets).sum().item()
    #             test_accuracy = 100. * test_correct / test_total
    #             print(f"[Test Phase] Epoch: {epoch+1} | Test Accuracy: {test_accuracy:.2f}%")
    #         scheduler.step()

    #     print("Saving pre-trained model...")
    #     torch.save({'backbone': net.state_dict()}, os.path.join(checkpoint_dir, save_path))
    #     return

# def test(load_path, length, num, words, feature_dim=512):
#     print("===============evaluation on model %s===============" % load_path)
#     num_classes = len(trainset.classes)
#     num_classes_test = len(testset.classes)
#     print("number of train identities: ", num_classes)
#     print("number of test identities: ", num_classes_test)
#     print("number of training images: ", len(trainset))
#     print("number of test images: ", len(testset))
#     print("number of training batches per epoch:", len(train_loader))
#     print("number of testing batches per epoch:", len(test_loader))

#     if args.cross_dataset:
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     else:
#         if args.dataset in ["facescrub", "cfw", "youtube"]:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
#         else:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim)

#     net = nn.DataParallel(net).to(device)

#     # <<< THAY ĐỔI BẮT ĐẦU >>>
#     # Kiểm tra nếu là đường dẫn tuyệt đối (ví dụ: /kaggle/input/...)
#     if os.path.isabs(load_path):
#         checkpoint_path = load_path
#     # Nếu không, dùng đường dẫn tương đối trong thư mục checkpoint
#     else:
#         checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#         checkpoint_path = os.path.join(checkpoint_dir, load_path)

#     # Kiểm tra xem file có tồn tại không trước khi load
#     if not os.path.exists(checkpoint_path):
#         print(f"Lỗi: Không tìm thấy file checkpoint tại: {checkpoint_path}")
#         sys.exit(1)
        
#     print(f"Loading weights for evaluation from {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path)
#     # <<< THAY ĐỔI KẾT THÚC >>>
    
#     net.load_state_dict(checkpoint['backbone'])
#     mlp_weight = checkpoint['mlp']
#     len_word = int(feature_dim / num)
#     net.eval()
#     with torch.no_grad():
#         index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
#         start = datetime.now()
#         query_features, test_labels = compute_quant(transform_test, test_loader, net, device)
#         if args.dataset != "vggface2":
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=5)
#         else:
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=10)

#         time_elapsed = datetime.now() - start
#         print("Query completed in %d ms" % int(time_elapsed.total_seconds() * 1000))
#         print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))



# def test(load_path, length, num, words, feature_dim=512):
#     print(f"=============== Evaluation on model {load_path} ===============")
#     num_classes = len(trainset.classes)
#     num_classes_test = len(testset.classes)
#     print(f"Number of train identities: {num_classes}")
#     print(f"Number of test identities: {num_classes_test}")
#     print(f"Number of training images: {len(trainset)}")
#     print(f"Number of test images: {len(testset)}")
#     print(f"Number of training batches per epoch: {len(train_loader)}")
#     print(f"Number of testing batches per epoch: {len(test_loader)}")

#     if args.cross_dataset:
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     else:
#         if args.dataset in ["facescrub", "cfw", "youtube"]:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
#         else:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim)

#     net = nn.DataParallel(net).to(device)

#     # Kiểm tra nếu là đường dẫn tuyệt đối (ví dụ: /kaggle/input/...)
#     if os.path.isabs(load_path):
#         checkpoint_path = load_path
#     else:
#         checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#         checkpoint_path = os.path.join(checkpoint_dir, load_path)

#     # Kiểm tra xem file có tồn tại không trước khi load
#     if not os.path.exists(checkpoint_path):
#         print(f"Error: Checkpoint file {checkpoint_path} not found")
#         sys.exit(1)
        
#     print(f"Loading pretrained weights from {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint['backbone'])
#     mlp_weight = checkpoint.get('mlp', None)  # Sử dụng get để tránh lỗi nếu 'mlp' không tồn tại

#     len_word = int(feature_dim / num)
#     net.eval()
    
#     # Tính thời gian truy vấn
#     total_query_time = 0
#     num_queries = len(testset)
    
#     with torch.no_grad():
#         # Tính index cho tập train
#         index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
        
#         # Đo thời gian truy vấn cho tập test
#         query_features, test_labels = compute_quant(transform_test, test_loader, net, device)
#         start = time.time()
#         if args.dataset != "vggface2":
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=5)
#         else:
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=10)
#         query_time = time.time() - start
#         total_query_time = query_time * 1000  # Chuyển sang ms
#         avg_query_time = total_query_time / num_queries  # ms/query
    
#     print(f"Query completed in {total_query_time:.2f} ms")
#     print(f"Average query time: {avg_query_time:.4f} ms/query")
#     print(f"[Evaluate Phase] mAP: {100. * float(mAP):.2f}% top_k: {100. * float(top_k):.2f}%")
    
# Có thể xuất ra được Top-K từ 10 đến 100 rồi, nhưng chưa fix lại phần tính thời gian và FLOP
# def test(load_path, length, num, words, feature_dim=512):
#     len_bit = int(num * math.log(words, 2))
#     assert length == len_bit, "something went wrong with code length"

#     print(f"=============== Evaluation on model {load_path} ===============")
#     num_classes = len(trainset.classes)
#     num_classes_test = len(testset.classes)
#     print(f"Number of train identities: {num_classes}")
#     print(f"Number of test identities: {num_classes_test}")
#     print(f"Number of training images: {len(trainset)}")
#     print(f"Number of test images: {len(testset)}")
#     print(f"Number of training batches per epoch: {len(train_loader)}")
#     print(f"Number of testing batches per epoch: {len(test_loader)}")

#     if args.cross_dataset:
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     else:
#         if args.dataset in ["facescrub", "cfw", "youtube"]:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
#         else:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim)

#     net = nn.DataParallel(net).to(device)

#     # Kiểm tra nếu là đường dẫn tuyệt đối (ví dụ: /kaggle/input/...)
#     if os.path.isabs(load_path):
#         checkpoint_path = load_path
#     else:
#         checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#         checkpoint_path = os.path.join(checkpoint_dir, load_path)

#     # Kiểm tra xem file có tồn tại không trước khi load
#     if not os.path.exists(checkpoint_path):
#         print(f"Error: Checkpoint file {checkpoint_path} not found")
#         sys.exit(1)
        
#     print(f"Loading pretrained weights from {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint['backbone'])
#     mlp_weight = checkpoint.get('mlp', None)  # Sử dụng get để tránh lỗi nếu 'mlp' không tồn tại

#     len_word = int(feature_dim / num)
#     net.eval()
    
#     # Tính thời gian truy vấn
#     total_query_time = 0
#     num_queries = len(testset)
    
#     with torch.no_grad():
#         # Tính index cho tập train
#         index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
        
#         # Đo thời gian truy vấn cho tập test
#         start = time.perf_counter()  # Sử dụng perf_counter để đo chính xác hơn
#         query_features, test_labels = compute_quant(transform_test, test_loader, net, device)
#         # Tính mAP một lần trên toàn bộ ranked list
#         mAP, _ = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=len(trainset))
#         print(f"[Evaluate Phase] mAP: {100. * float(mAP):.2f}%")
#         # Vòng lặp cho top-k từ 10 đến 100, step 10, chỉ tính top-k accuracy
#         for k in range(10, 101, 10):
#             _, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=k)
#             print(f"[Evaluate Phase @ top-{k}] top_k: {100. * float(top_k):.2f}%")
#         total_query_time = (time.perf_counter() - start) * 1000  # Chuyển sang ms
#         avg_query_time = total_query_time / num_queries  # ms/query
    
#     print(f"Query completed in {total_query_time:.2f} ms")
#     print(f"Average query time: {avg_query_time:.4f} ms/query")



#Code lol què
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# from datetime import datetime
# import torch.distributions as Distributions
# import math
# import argparse
# import sys
# import time
# import os
# from torch.optim.lr_scheduler import LambdaLR
# from utils import Logger, AverageMeter, compute_quant, compute_quant_indexing, PqDistRet_Ortho, PqDistRet_Ortho_safe
# from backbone import resnet20_pq, SphereNet20_pq, EdgeFaceBackbone
# from margin_metric import OrthoPQ, CosFace
# from data_loader import get_datasets_transform
# from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR  # Thêm import ReduceLROnPlateau


# parser = argparse.ArgumentParser(description='PyTorch Implementation of Orthonormal Product Quantization for Scalable Face Image Retrieval')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate mode turned on')
# parser.add_argument('-c', '--cross-dataset', action='store_true', help='generalize on unseen identities')
# parser.add_argument('--bs', type=int, default=256, help='Batch size of each iteration')
# parser.add_argument('--save', nargs='+', help='path to saving models, accept multiple arguments as list')
# parser.add_argument('--load', nargs='+', help='path to loading models, accept multiple arguments as list')
# parser.add_argument('--len', nargs='+', type=int, help='length of hashing codes, accept multiple arguments as list')
# parser.add_argument('--dataset', type=str, default='facescrub', help='which dataset for training (one of facescrub, youtube, CFW, and VGGFace2)')
# parser.add_argument('--num', nargs='+', type=int, help='num. of codebooks, could be 4, 8...')
# parser.add_argument('--words', nargs='+', type=int, default=[256, 256, 256, 256], help='num of words, should be exponential of 2')
# parser.add_argument('--margin', default=0.4, type=float, help='margin of cosine similarity')
# parser.add_argument('--miu', default=0.1, type=float, help='Balance weight of redundancy loss')
# parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'edgeface'], help='Backbone type: resnet or edgeface')
# parser.add_argument('--data_dir', type=str, default='/kaggle/input/facescrub-0210-3', help='Data direction on kaggle for multiple dataset')
# parser.add_argument('--sc', default=30, type=float, help='scale s for initialize metric')
# parser.add_argument('--pretrain_cosface', action='store_true', help='Pretrain with CosFace loss before OrthoPQ')
# parser.add_argument('--s_cosface', default=30.0, type=float, help='scale s for CosFace')  # Thêm
# parser.add_argument('--m_cosface', default=0.2, type=float, help='margin m for CosFace')  # Thêm
# parser.add_argument('--max_norm', default=0.5, type=float, help='gradient clipping max norm for pre-train')  # Thêm
# parser.add_argument('--epochs_cosface', default=50, type=int, help='number of epochs for CosFace pre-training')  # Thêm
# parser.add_argument('--lr_backbone', default=0.0001, type=float, help='learning rate for backbone in pre-train CosFace')  # Thêm


# try:
#     args = parser.parse_args()
# except Exception as e:
#     print(f"Parser error: {e}")
#     sys.exit(1)

# trainset, testset = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['dataset']
# transform_train, transform_test = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['transform']

# train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=4)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=4)

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# torch.cuda.manual_seed_all(1)

# def train(save_path, length, num, words, feature_dim):
#     best_acc = 0
#     best_mAP = 0
#     best_epoch = 1
#     print('==> Building model..')
#     num_classes = len(trainset.classes)
#     print("number of identities: ", num_classes)
#     print("number of training images: ", len(trainset))
#     print("number of test images: ", len(testset))
#     print("number of training batches per epoch:", len(train_loader))
#     print("number of testing batches per epoch:", len(test_loader))

#     if args.cross_dataset or args.dataset == "vggface2":
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     else:
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)

#     net = nn.DataParallel(net).to(device)
#     cudnn.benchmark = True

#     # if args.pretrain_cosface:
#     #     print("Pre-training with CosFace loss...")
#     #     metric = CosFace(in_features=feature_dim, out_features=num_classes, s=30.0, m=0.4)
#     #     metric = nn.DataParallel(metric).to(device)
#     #     criterion = nn.CrossEntropyLoss()
#     #     optimizer = optim.AdamW([
#     #         {'params': net.parameters(), 'lr': args.lr},  # lr=0.000001
#     #         {'params': metric.parameters(), 'lr': args.lr * 10}  # lr=0.001
#     #     ], weight_decay=5e-4)
#     #     def poly_decay_with_restarts(epoch):
#     #         base_lr = 1.0
#     #         decay = (1 - (epoch % 10) / 10) ** 0.9  # Decay mỗi 10 epoch
#     #         return base_lr * decay
#     #     #scheduler = LambdaLR(optimizer, lr_lambda=poly_decay_with_restarts)
#     #     scheduler = CosineAnnealingLR(optimizer, T_max=50)
#     #     checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#     #     os.makedirs(checkpoint_dir, exist_ok=True)

#     #     for epoch in range(60):
#     #         net.train()
#     #         metric.train()
#     #         losses = AverageMeter()
#     #         grad_norm_backbone = 0
#     #         grad_norm_metric = 0
#     #         correct = 0
#     #         total = 0
#     #         start = time.time()
#     #         for batch_idx, (inputs, targets) in enumerate(train_loader):
#     #             inputs, targets = inputs.to(device), targets.to(device)
#     #             transformed_images = transform_train(inputs)
#     #             features = net(transformed_images)
#     #             outputs = metric(features, targets)
#     #             loss = criterion(outputs, targets)
#     #             optimizer.zero_grad()
#     #             loss.backward()
#     #             grad_norm_b = torch.norm(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])).item()
#     #             grad_norm_m = torch.norm(torch.cat([p.grad.flatten() for p in metric.parameters() if p.grad is not None])).item()
#     #             torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping
#     #             optimizer.step()
#     #             losses.update(loss.item(), len(inputs))
#     #             grad_norm_backbone += grad_norm_b
#     #             grad_norm_metric += grad_norm_m
#     #             _, predicted = outputs.max(1)
#     #             total += targets.size(0)
#     #             correct += predicted.eq(targets).sum().item()

#     #         # In log trung bình mỗi epoch
#     #         avg_loss = losses.avg
#     #         avg_grad_norm_backbone = grad_norm_backbone / len(train_loader)
#     #         avg_grad_norm_metric = grad_norm_metric / len(train_loader)
#     #         accuracy = 100. * correct / total
#     #         epoch_elapsed = time.time() - start
#     #         print(f"Pre-train Epoch {epoch+1} | Loss: {avg_loss:.4f} | Grad_norm_backbone: {avg_grad_norm_backbone:.4f} | Grad_norm_metric: {avg_grad_norm_metric:.4f} | Accuracy: {accuracy:.2f}%")

#     #         # Đánh giá test accuracy mỗi 5 epoch
#     #         if (epoch + 1) % 5 == 0:
#     #             net.eval()
#     #             metric.eval()
#     #             test_correct = 0
#     #             test_total = 0
#     #             with torch.no_grad():
#     #                 for inputs, targets in test_loader:
#     #                     inputs, targets = inputs.to(device), targets.to(device)
#     #                     features = net(transform_test(inputs))
#     #                     outputs = metric(features, targets)
#     #                     _, predicted = outputs.max(1)
#     #                     test_total += targets.size(0)
#     #                     test_correct += predicted.eq(targets).sum().item()
#     #             test_accuracy = 100. * test_correct / test_total
#     #             print(f"[Test Phase] Epoch: {epoch+1} | Test Accuracy: {test_accuracy:.2f}%")

#     #         scheduler.step()

#     #     print("Saving pre-trained model...")
#     #     torch.save({'backbone': net.state_dict()}, os.path.join(checkpoint_dir, save_path))
#     #     return

#     if args.pretrain_cosface:
#         print("Pre-training with CosFace loss...")
#         metric = CosFace(in_features=feature_dim, out_features=num_classes, s=args.s_cosface, m=args.m_cosface)  # Dùng s, m từ argparse
#         metric = nn.DataParallel(metric).to(device)
#         criterion = nn.CrossEntropyLoss()
#         # Freeze conv1 và layer1 để bảo vệ weights pretrained
#         for name, param in net.named_parameters():
#             if 'conv1' in name or 'layer1' in name:
#                 param.requires_grad = False
#         optimizer = optim.AdamW([
#             {'params': [p for p in net.parameters() if p.requires_grad], 'lr': args.lr_backbone},  # Dùng lr_backbone
#             {'params': metric.parameters(), 'lr': args.lr_backbone * 10}  # lr metric = lr_backbone * 10
#         ], weight_decay=5e-4)
#         scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_cosface)  # Dùng epochs_cosface
#         checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#         os.makedirs(checkpoint_dir, exist_ok=True)

#         for epoch in range(args.epochs_cosface):  # Dùng epochs_cosface
#             net.train()
#             metric.train()
#             losses = AverageMeter()
#             grad_norm_backbone = 0
#             grad_norm_metric = 0
#             correct = 0
#             total = 0
#             start = time.time()
#             for batch_idx, (inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 transformed_images = transform_train(inputs)
#                 features = net(transformed_images)
#                 outputs = metric(features, targets)
#                 loss = criterion(outputs, targets)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 grad_norm_b = torch.norm(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])).item()
#                 grad_norm_m = torch.norm(torch.cat([p.grad.flatten() for p in metric.parameters() if p.grad is not None])).item()
#                 torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)  # Dùng max_norm
#                 torch.nn.utils.clip_grad_norm_(metric.parameters(), max_norm=args.max_norm)
#                 optimizer.step()
#                 losses.update(loss.item(), len(inputs))
#                 grad_norm_backbone += grad_norm_b
#                 grad_norm_metric += grad_norm_m
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()
#                 #print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Grad_norm_backbone: {grad_norm_b:.4f} | Grad_norm_metric: {grad_norm_m:.4f}")

#             avg_loss = losses.avg
#             avg_grad_norm_backbone = grad_norm_backbone / len(train_loader)
#             avg_grad_norm_metric = grad_norm_metric / len(train_loader)
#             accuracy = 100. * correct / total
#             epoch_elapsed = time.time() - start
#             print(f"Pre-train Epoch {epoch+1} | Loss: {avg_loss:.4f} | Grad_norm_backbone: {avg_grad_norm_backbone:.4f} | Grad_norm_metric: {avg_grad_norm_metric:.4f} | Accuracy: {accuracy:.2f}%")

#             if (epoch + 1) % 5 == 0:
#                 net.eval()
#                 metric.eval()
#                 test_correct = 0
#                 test_total = 0
#                 with torch.no_grad():
#                     for inputs, targets in test_loader:
#                         inputs, targets = inputs.to(device), targets.to(device)
#                         features = net(transform_test(inputs))
#                         outputs = metric(features, targets)
#                         _, predicted = outputs.max(1)
#                         test_total += targets.size(0)
#                         test_correct += predicted.eq(targets).sum().item()
#                 test_accuracy = 100. * test_correct / test_total
#                 print(f"[Test Phase] Epoch: {epoch+1} | Test Accuracy: {test_accuracy:.2f}%")

#             scheduler.step()

#     print("Saving pre-trained model...")
#     torch.save({'backbone': net.state_dict()}, os.path.join(checkpoint_dir, save_path))
#     return

#     if args.load:
#         checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#         checkpoint_path = os.path.join(checkpoint_dir, args.load[0])
#         checkpoint = torch.load(checkpoint_path)
#         net.load_state_dict(checkpoint['backbone'])
#         print(f"Loaded pretrained weights from {checkpoint_path}")

#     d = int(feature_dim / num)
#     matrix = torch.randn(d, d)
#     for k in range(d):
#         for j in range(d):
#             matrix[j, k] = math.cos((j+0.5)*k*math.pi/d)
#     matrix[:, 0] /= math.sqrt(2)
#     matrix /= math.sqrt(d/2)
#     code_books = torch.Tensor(num, d, words)
#     code_books[0] = matrix[:, :words]
#     for i in range(1, num):
#         code_books[i] = matrix @ code_books[i-1]
#     code_books /= torch.norm(code_books, dim=1, keepdim=True)
#     print("Codebook norms:", [torch.norm(code_books[i], dim=1).mean().item() for i in range(num)])

#     metric = OrthoPQ(in_features=feature_dim, out_features=num_classes, num_books=num, num_words=words, code_books=code_books, sc=args.sc, m=args.margin)
#     metric = nn.DataParallel(metric).to(device)

#     criterion = nn.CrossEntropyLoss()
#     num_books = num
#     num_words = words
#     len_word = int(feature_dim / num_books)
#     len_bit = int(num_books * math.log(num_words, 2))
#     assert length == len_bit, f"Code length mismatch: expected {length}-bit, got {len_bit}-bit"
#     print("num. of codebooks: ", num_books)
#     print("num. of words per book: ", num_words)
#     print("dim. of word: ", len_word)
#     print("code length: %d-bit \t learning rate: %.3f \t scale length: %d \t penalty margin: %.2f \t balance_weight: %.3f" % 
#           (len_bit, args.lr, metric.module.s, metric.module.m, args.miu))

#     optimizer_params = [
#         {'params': metric.parameters(), 'lr': args.lr},
#         {'params': [p for p in net.parameters() if p.requires_grad], 'lr': args.lr * 0.1}
#     ]
#     optimizer = optim.SGD(optimizer_params, weight_decay=1e-3, momentum=0.9)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
#     EPOCHS = 300 if args.dataset in ["facescrub", "cfw", "youtube"] else 160
#     # Epoch thay từ 200 lên 300 vì giảm lr từ 0.1 thành 0.005
#     since = time.time()
#     best_loss = 1e3

#     for epoch in range(EPOCHS):
#         print('==> Epoch: %d' % (epoch+1))
#         net.train()
#         metric.train()
#         losses = AverageMeter()
#         loss_clf_avg = AverageMeter()
#         loss_entropy_avg = AverageMeter()
#         grad_norm_backbone = 0
#         grad_norm_metric = 0
#         correct = 0
#         total = 0
#         start = time.time()
#         for batch_idx, (inputs, targets) in enumerate(train_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             transformed_images = transform_train(inputs)
#             features = net(transformed_images)
#             output1, output2, xc_probs = metric(features, targets)
#             loss_clf1 = [criterion(output1[:, i, :], targets) for i in range(num_books)]
#             loss_clf2 = [criterion(output2[:, i, :], targets) for i in range(num_books)]
#             loss_clf = 0.5 * (sum(loss_clf1) / len(loss_clf1) + sum(loss_clf2) / len(loss_clf2))
#             xc_entropy = [Distributions.categorical.Categorical(probs=xc_probs[:, i, :]).entropy().sum() for i in range(num_books)]
#             loss_entropy = sum(xc_entropy) / (num_books * len(inputs))
#             loss = loss_clf + args.miu * loss_entropy
#             optimizer.zero_grad()
#             loss.backward()
#             grad_norm_b = torch.norm(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])).item()
#             grad_norm_m = torch.norm(torch.cat([p.grad.flatten() for p in metric.parameters() if p.grad is not None])).item()
#             #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)  # Giảm max_norm từ 1.0 thành 0.5
#             torch.nn.utils.clip_grad_norm_(metric.parameters(), max_norm=args.max_norm)
#             optimizer.step()
#             losses.update(loss.item(), len(inputs))
#             loss_clf_avg.update(loss_clf.item(), len(inputs))
#             loss_entropy_avg.update(loss_entropy.item(), len(inputs))
#             grad_norm_backbone += grad_norm_b
#             grad_norm_metric += grad_norm_m
#             _, predicted = output1[:, 0, :].max(1)  # Dùng output1 của codebook đầu để tính accuracy
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#         # In log trung bình mỗi epoch
#         avg_loss = losses.avg
#         avg_loss_clf = loss_clf_avg.avg
#         avg_loss_entropy = loss_entropy_avg.avg
#         avg_grad_norm_backbone = grad_norm_backbone / len(train_loader)
#         avg_grad_norm_metric = grad_norm_metric / len(train_loader)
#         accuracy = 100. * correct / total
#         epoch_elapsed = time.time() - start
#         print(f'Epoch: {epoch+1} | Loss_clf: {avg_loss_clf:.4f} | Loss_entropy: {avg_loss_entropy:.4f} | Total Loss: {avg_loss:.4f} | Grad_norm_backbone: {avg_grad_norm_backbone:.4f} | Grad_norm_metric: {avg_grad_norm_metric:.4f} | Accuracy: {accuracy:.2f}%')
#         print("Epoch Completed in {:.0f}min {:.0f}s".format(epoch_elapsed // 60, epoch_elapsed % 60))
#         scheduler.step(avg_loss)

#         if (epoch+1) % 5 == 0:
#             net.eval()
#             with torch.no_grad():
#                 mlp_weight = metric.module.mlp
#                 index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
#                 queries, test_labels = compute_quant(transform_test, test_loader, net, device)
#                 start = time.time()
#                 mAP, top_k = PqDistRet_Ortho(queries, test_labels, train_labels, index, mlp_weight, len_word, num_books, device, top=50)
#                 time_elapsed = time.time() - start
#                 print("Code generated in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#                 print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 best_mAP = mAP
#                 print('Saving..')
#                 checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#                 os.makedirs(checkpoint_dir, exist_ok=True)
#                 torch.save({'backbone': net.state_dict(), 'mlp': metric.module.mlp}, os.path.join(checkpoint_dir, save_path))
#                 best_epoch = epoch + 1
#     time_elapsed = time.time() - since
#     print("Training Completed in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#     print("Best mAP {:.4f} at epoch {}".format(best_mAP, best_epoch))
#     print("Model saved as %s" % save_path)

# def test(load_path, length, num, words, feature_dim=512):
#     print("===============evaluation on model %s===============" % load_path)
#     num_classes = len(trainset.classes)
#     num_classes_test = len(testset.classes)
#     print("number of train identities: ", num_classes)
#     print("number of test identities: ", num_classes_test)
#     print("number of training images: ", len(trainset))
#     print("number of test images: ", len(testset))
#     print("number of training batches per epoch:", len(train_loader))
#     print("number of testing batches per epoch:", len(test_loader))

#     if args.cross_dataset:
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     else:
#         if args.dataset in ["facescrub", "cfw", "youtube"]:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
#         else:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim)

#     net = nn.DataParallel(net).to(device)

#     checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#     checkpoint_path = os.path.join(checkpoint_dir, load_path)
#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint['backbone'])
#     mlp_weight = checkpoint['mlp']
#     len_word = int(feature_dim / num)
#     net.eval()
#     with torch.no_grad():
#         index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
#         start = datetime.now()
#         query_features, test_labels = compute_quant(transform_test, test_loader, net, device)
#         if args.dataset != "vggface2":
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=5)
#         else:
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=10)

#         time_elapsed = datetime.now() - start
#         print("Query completed in %d ms" % int(time_elapsed.total_seconds() * 1000))
#         print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

# if __name__ == "__main__":
#     save_dir = 'log'
#     if args.evaluate:
#         if not args.load:
#             print("Error: --load is required for evaluation mode")
#             sys.exit(1)
#         if len(args.load) != len(args.num) or len(args.load) != len(args.len) or len(args.load) != len(args.words):
#             print("Warning: Args lengths don't match. Adjusting to shortest length.")
#             min_len = min(len(args.load), len(args.num), len(args.len), len(args.words))
#             args.load = args.load[:min_len]
#             args.num = args.num[:min_len]
#             args.len = args.len[:min_len]
#             args.words = args.words[:min_len]
#         for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
#             if args.cross_dataset:
#                 feature_dim = num_s * words_s
#             else:
#                 if args.dataset != "vggface2":
#                     if args.len[i] != 36:
#                         feature_dim = 512
#                     else:
#                         feature_dim = 516
#                 else:
#                     feature_dim = num_s * words_s
#             test(args.load[i], args.len[i], num_s, words_s, feature_dim=feature_dim)
#     else:
#         if not args.save:
#             print("Error: --save is required for training mode")
#             sys.exit(1)
#         # if args.pretrain_cosface:
#         #     sys.stdout = Logger(os.path.join(save_dir,
#         #         'cosface_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
#         #     print("[Configuration] Pre-training on dataset: %s\n Batch_size: %d\n learning rate backbone: %.6f\n learning rate metric: %.6f" %
#         #           (args.dataset, args.bs, args.lr * 0.01, args.lr * 10))
#         #     train(args.save[0], None, None, None, feature_dim=512)
#         if args.pretrain_cosface:
#             sys.stdout = Logger(os.path.join(save_dir,
#                 'cosface_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
#             print("[Configuration] Pre-training on dataset: %s\n Batch_size: %d\n learning rate backbone: %.6f\n learning rate metric: %.6f\n s: %.1f\n m: %.1f\n max_norm: %.1f\n epochs: %d" %
#                 (args.dataset, args.bs, args.lr_backbone, args.lr_backbone * 10, args.s_cosface, args.m_cosface, args.max_norm, args.epochs_cosface))
#             train(args.save[0], None, None, None, feature_dim=512)
#         else:
#             if len(args.save) != len(args.num) or len(args.save) != len(args.len) or len(args.save) != len(args.words):
#                 print("Warning: Args lengths don't match. Adjusting to shortest length.")
#                 min_len = min(len(args.save), len(args.num), len(args.len), len(args.words))
#                 args.save = args.save[:min_len]
#                 args.num = args.num[:min_len]
#                 args.len = args.len[:min_len]
#                 args.words = args.words[:min_len]
#             for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
#                 sys.stdout = Logger(os.path.join(save_dir,
#                     str(args.len[i]) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
#                 print("[Configuration] Training on dataset: %s\n Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n num_books: %d\n num_words: %d" %
#                       (args.dataset, args.len[i], args.bs, args.lr, num_s, words_s))
#                 print("HyperParams:\nmargin: %.3f\t miu: %.4f" % (args.margin, args.miu))
#                 if args.dataset != "vggface2":
#                     if args.len[i] != 36:
#                         feature_dim = 512
#                     else:
#                         feature_dim = 516
#                 else:
#                     feature_dim = num_s * words_s
#                 train(args.save[i], args.len[i], num_s, words_s, feature_dim=feature_dim)






# Nhức đầu quá chưa biết phải làm gì
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# from datetime import datetime
# import torch.distributions as Distributions
# import math
# import argparse
# import sys
# import time
# import os
# from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
# from utils import Logger, AverageMeter, compute_quant, compute_quant_indexing, PqDistRet_Ortho, PqDistRet_Ortho_safe
# from backbone import resnet20_pq, SphereNet20_pq, EdgeFaceBackbone
# from margin_metric import OrthoPQ, CosFace
# from data_loader import get_datasets_transform

# parser = argparse.ArgumentParser(description='PyTorch Implementation of Orthonormal Product Quantization for Scalable Face Image Retrieval')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate mode turned on')
# parser.add_argument('-c', '--cross-dataset', action='store_true', help='generalize on unseen identities')
# parser.add_argument('--bs', type=int, default=256, help='Batch size of each iteration')
# parser.add_argument('--save', nargs='+', help='path to saving models, accept multiple arguments as list')
# parser.add_argument('--load', nargs='+', help='path to loading models, accept multiple arguments as list')
# parser.add_argument('--len', nargs='+', type=int, help='length of hashing codes, accept multiple arguments as list')
# parser.add_argument('--dataset', type=str, default='facescrub', help='which dataset for training (one of facescrub, youtube, CFW, and VGGFace2)')
# parser.add_argument('--num', nargs='+', type=int, help='num. of codebooks, could be 4, 8...')
# parser.add_argument('--words', nargs='+', type=int, default=[256, 256, 256, 256], help='num of words, should be exponential of 2')
# parser.add_argument('--margin', default=0.4, type=float, help='margin of cosine similarity')
# parser.add_argument('--miu', default=0.1, type=float, help='Balance weight of redundancy loss')
# parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'edgeface'], help='Backbone type: resnet or edgeface')
# parser.add_argument('--data_dir', type=str, default='/kaggle/input/facescrub-0210-3', help='Data direction on kaggle for multiple dataset')
# parser.add_argument('--sc', default=30, type=float, help='scale s for initialize metric')
# parser.add_argument('--pretrain_cosface', action='store_true', help='Pretrain with CosFace loss before OrthoPQ')

# try:
#     args = parser.parse_args()
# except Exception as e:
#     print(f"Parser error: {e}")
#     sys.exit(1)

# trainset, testset = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['dataset']
# transform_train, transform_test = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['transform']

# train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=4)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=4)

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# torch.cuda.manual_seed_all(1)

# def train(save_path, length, num, words, feature_dim):
#     best_acc = 0
#     best_mAP = 0
#     best_epoch = 1
#     print('==> Building model..')
#     num_classes = len(trainset.classes)
#     print("number of identities: ", num_classes)
#     print("number of training images: ", len(trainset))
#     print("number of test images: ", len(testset))
#     print("number of training batches per epoch:", len(train_loader))
#     print("number of testing batches per epoch:", len(test_loader))

#     if args.cross_dataset or args.dataset == "vggface2":
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     else:
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)

#     net = nn.DataParallel(net).to(device)
#     cudnn.benchmark = True

#     if args.pretrain_cosface:
#         print("Pre-training with CosFace loss...")
#         metric = CosFace(in_features=feature_dim, out_features=num_classes, s=30.0, m=0.4)
#         metric = nn.DataParallel(metric).to(device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.AdamW([
#             {'params': net.parameters(), 'lr': args.lr * 0.01},  # lr=0.000001
#             {'params': metric.parameters(), 'lr': args.lr * 10}  # lr=0.001
#         ], weight_decay=5e-4)
#         scheduler = CosineAnnealingLR(optimizer, T_max=50)
#         checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#         os.makedirs(checkpoint_dir, exist_ok=True)

#         for epoch in range(50):
#             net.train()
#             metric.train()
#             losses = AverageMeter()
#             grad_norm_backbone = 0
#             grad_norm_metric = 0
#             correct = 0
#             total = 0
#             start = time.time()
#             for batch_idx, (inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 transformed_images = transform_train(inputs)
#                 features = net(transformed_images)
#                 outputs = metric(features, targets)
#                 loss = criterion(outputs, targets)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 grad_norm_b = torch.norm(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])).item()
#                 grad_norm_m = torch.norm(torch.cat([p.grad.flatten() for p in metric.parameters() if p.grad is not None])).item()
#                 torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping
#                 optimizer.step()
#                 losses.update(loss.item(), len(inputs))
#                 grad_norm_backbone += grad_norm_b
#                 grad_norm_metric += grad_norm_m
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()

#             # In log trung bình mỗi epoch
#             avg_loss = losses.avg
#             avg_grad_norm_backbone = grad_norm_backbone / len(train_loader)
#             avg_grad_norm_metric = grad_norm_metric / len(train_loader)
#             accuracy = 100. * correct / total
#             epoch_elapsed = time.time() - start
#             print(f"Pre-train Epoch {epoch+1} | Loss: {avg_loss:.4f} | Grad_norm_backbone: {avg_grad_norm_backbone:.4f} | Grad_norm_metric: {avg_grad_norm_metric:.4f} | Accuracy: {accuracy:.2f}%")

#             # Đánh giá test accuracy mỗi 5 epoch
#             if (epoch + 1) % 5 == 0:
#                 net.eval()
#                 metric.eval()
#                 test_correct = 0
#                 test_total = 0
#                 with torch.no_grad():
#                     for inputs, targets in test_loader:
#                         inputs, targets = inputs.to(device), targets.to(device)
#                         features = net(transform_test(inputs))
#                         outputs = metric(features, targets)
#                         _, predicted = outputs.max(1)
#                         test_total += targets.size(0)
#                         test_correct += predicted.eq(targets).sum().item()
#                 test_accuracy = 100. * test_correct / test_total
#                 print(f"[Test Phase] Epoch: {epoch+1} | Test Accuracy: {test_accuracy:.2f}%")

#             scheduler.step()

#         print("Saving pre-trained model...")
#         torch.save({'backbone': net.state_dict()}, os.path.join(checkpoint_dir, save_path))
#         return

#     if args.load:
#         checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#         checkpoint_path = os.path.join(checkpoint_dir, args.load[0])
#         checkpoint = torch.load(checkpoint_path)
#         net.load_state_dict(checkpoint['backbone'])
#         print(f"Loaded pretrained weights from {checkpoint_path}")

#     d = int(feature_dim / num)
#     matrix = torch.randn(d, d)
#     for k in range(d):
#         for j in range(d):
#             matrix[j, k] = math.cos((j+0.5)*k*math.pi/d)
#     matrix[:, 0] /= math.sqrt(2)
#     matrix /= math.sqrt(d/2)
#     code_books = torch.Tensor(num, d, words)
#     code_books[0] = matrix[:, :words]
#     for i in range(1, num):
#         code_books[i] = matrix @ code_books[i-1]
#     code_books /= torch.norm(code_books, dim=1, keepdim=True)
#     print("Codebook norms:", [torch.norm(code_books[i], dim=1).mean().item() for i in range(num)])

#     metric = OrthoPQ(in_features=feature_dim, out_features=num_classes, num_books=num, num_words=words, code_books=code_books, sc=args.sc, m=args.margin)
#     metric = nn.DataParallel(metric).to(device)

#     criterion = nn.CrossEntropyLoss()
#     num_books = num
#     num_words = words
#     len_word = int(feature_dim / num_books)
#     len_bit = int(num_books * math.log(num_words, 2))
#     assert length == len_bit, f"Code length mismatch: expected {length}-bit, got {len_bit}-bit"
#     print("num. of codebooks: ", num_books)
#     print("num. of words per book: ", num_words)
#     print("dim. of word: ", len_word)
#     print("code length: %d-bit \t learning rate: %.3f \t scale length: %d \t penalty margin: %.2f \t balance_weight: %.3f" % 
#           (len_bit, args.lr, metric.module.s, metric.module.m, args.miu))

#     optimizer_params = [
#         {'params': metric.parameters(), 'lr': args.lr},
#         {'params': [p for p in net.parameters() if p.requires_grad], 'lr': args.lr * 0.1}
#     ]
#     optimizer = optim.SGD(optimizer_params, weight_decay=1e-3, momentum=0.9)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
#     EPOCHS = 200 if args.dataset in ["facescrub", "cfw", "youtube"] else 160

#     since = time.time()
#     best_loss = 1e3

#     for epoch in range(EPOCHS):
#         print('==> Epoch: %d' % (epoch+1))
#         net.train()
#         metric.train()
#         losses = AverageMeter()
#         loss_clf_avg = AverageMeter()
#         loss_entropy_avg = AverageMeter()
#         grad_norm_backbone = 0
#         grad_norm_metric = 0
#         correct = 0
#         total = 0
#         start = time.time()
#         for batch_idx, (inputs, targets) in enumerate(train_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             transformed_images = transform_train(inputs)
#             features = net(transformed_images)
#             output1, output2, xc_probs = metric(features, targets)
#             loss_clf1 = [criterion(output1[:, i, :], targets) for i in range(num_books)]
#             loss_clf2 = [criterion(output2[:, i, :], targets) for i in range(num_books)]
#             loss_clf = 0.5 * (sum(loss_clf1) / len(loss_clf1) + sum(loss_clf2) / len(loss_clf2))
#             xc_entropy = [Distributions.categorical.Categorical(probs=xc_probs[:, i, :]).entropy().sum() for i in range(num_books)]
#             loss_entropy = sum(xc_entropy) / (num_books * len(inputs))
#             loss = loss_clf + args.miu * loss_entropy
#             optimizer.zero_grad()
#             loss.backward()
#             grad_norm_b = torch.norm(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])).item()
#             grad_norm_m = torch.norm(torch.cat([p.grad.flatten() for p in metric.parameters() if p.grad is not None])).item()
#             torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping
#             optimizer.step()
#             losses.update(loss.item(), len(inputs))
#             loss_clf_avg.update(loss_clf.item(), len(inputs))
#             loss_entropy_avg.update(loss_entropy.item(), len(inputs))
#             grad_norm_backbone += grad_norm_b
#             grad_norm_metric += grad_norm_m
#             _, predicted = output1[:, 0, :].max(1)  # Dùng output1 của codebook đầu để tính accuracy
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#         # In log trung bình mỗi epoch
#         avg_loss = losses.avg
#         avg_loss_clf = loss_clf_avg.avg
#         avg_loss_entropy = loss_entropy_avg.avg
#         avg_grad_norm_backbone = grad_norm_backbone / len(train_loader)
#         avg_grad_norm_metric = grad_norm_metric / len(train_loader)
#         accuracy = 100. * correct / total
#         epoch_elapsed = time.time() - start
#         print(f'Epoch: {epoch+1} | Loss_clf: {avg_loss_clf:.4f} | Loss_entropy: {avg_loss_entropy:.4f} | Total Loss: {avg_loss:.4f} | Grad_norm_backbone: {avg_grad_norm_backbone:.4f} | Grad_norm_metric: {avg_grad_norm_metric:.4f} | Accuracy: {accuracy:.2f}%')
#         print("Epoch Completed in {:.0f}min {:.0f}s".format(epoch_elapsed // 60, epoch_elapsed % 60))
#         scheduler.step(avg_loss)

#         if (epoch+1) % 5 == 0:
#             net.eval()
#             with torch.no_grad():
#                 mlp_weight = metric.module.mlp
#                 index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
#                 queries, test_labels = compute_quant(transform_test, test_loader, net, device)
#                 start = time.time()
#                 mAP, top_k = PqDistRet_Ortho(queries, test_labels, train_labels, index, mlp_weight, len_word, num_books, device, top=50)
#                 time_elapsed = time.time() - start
#                 print("Code generated in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#                 print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 best_mAP = mAP
#                 print('Saving..')
#                 checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#                 os.makedirs(checkpoint_dir, exist_ok=True)
#                 torch.save({'backbone': net.state_dict(), 'mlp': metric.module.mlp}, os.path.join(checkpoint_dir, save_path))
#                 best_epoch = epoch + 1
#     time_elapsed = time.time() - since
#     print("Training Completed in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#     print("Best mAP {:.4f} at epoch {}".format(best_mAP, best_epoch))
#     print("Model saved as %s" % save_path)

# def test(load_path, length, num, words, feature_dim=512):
#     print("===============evaluation on model %s===============" % load_path)
#     num_classes = len(trainset.classes)
#     num_classes_test = len(testset.classes)
#     print("number of train identities: ", num_classes)
#     print("number of test identities: ", num_classes_test)
#     print("number of training images: ", len(trainset))
#     print("number of test images: ", len(testset))
#     print("number of training batches per epoch:", len(train_loader))
#     print("number of testing batches per epoch:", len(test_loader))

#     if args.cross_dataset:
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     else:
#         if args.dataset in ["facescrub", "cfw", "youtube"]:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
#         else:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim)

#     net = nn.DataParallel(net).to(device)

#     checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#     checkpoint_path = os.path.join(checkpoint_dir, load_path)
#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint['backbone'])
#     mlp_weight = checkpoint['mlp']
#     len_word = int(feature_dim / num)
#     net.eval()
#     with torch.no_grad():
#         index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
#         start = datetime.now()
#         query_features, test_labels = compute_quant(transform_test, test_loader, net, device)
#         if args.dataset != "vggface2":
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=5)
#         else:
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=10)

#         time_elapsed = datetime.now() - start
#         print("Query completed in %d ms" % int(time_elapsed.total_seconds() * 1000))
#         print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

# if __name__ == "__main__":
#     save_dir = 'log'
#     if args.evaluate:
#         if not args.load:
#             print("Error: --load is required for evaluation mode")
#             sys.exit(1)
#         if len(args.load) != len(args.num) or len(args.load) != len(args.len) or len(args.load) != len(args.words):
#             print("Warning: Args lengths don't match. Adjusting to shortest length.")
#             min_len = min(len(args.load), len(args.num), len(args.len), len(args.words))
#             args.load = args.load[:min_len]
#             args.num = args.num[:min_len]
#             args.len = args.len[:min_len]
#             args.words = args.words[:min_len]
#         for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
#             if args.cross_dataset:
#                 feature_dim = num_s * words_s
#             else:
#                 if args.dataset != "vggface2":
#                     if args.len[i] != 36:
#                         feature_dim = 512
#                     else:
#                         feature_dim = 516
#                 else:
#                     feature_dim = num_s * words_s
#             test(args.load[i], args.len[i], num_s, words_s, feature_dim=feature_dim)
#     else:
#         if not args.save:
#             print("Error: --save is required for training mode")
#             sys.exit(1)
#         if args.pretrain_cosface:
#             sys.stdout = Logger(os.path.join(save_dir,
#                 'cosface_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
#             print("[Configuration] Pre-training on dataset: %s\n Batch_size: %d\n learning rate: %.6f" %
#                   (args.dataset, args.bs, args.lr))
#             train(args.save[0], None, None, None, feature_dim=512)
#         else:
#             if len(args.save) != len(args.num) or len(args.save) != len(args.len) or len(args.save) != len(args.words):
#                 print("Warning: Args lengths don't match. Adjusting to shortest length.")
#                 min_len = min(len(args.save), len(args.num), len(args.len), len(args.words))
#                 args.save = args.save[:min_len]
#                 args.num = args.num[:min_len]
#                 args.len = args.len[:min_len]
#                 args.words = args.words[:min_len]
#             for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
#                 sys.stdout = Logger(os.path.join(save_dir,
#                     str(args.len[i]) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
#                 print("[Configuration] Training on dataset: %s\n  Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n num_books: %d\n num_words: %d" %
#                       (args.dataset, args.len[i], args.bs, args.lr, num_s, words_s))
#                 print("HyperParams:\nmargin: %.3f\t miu: %.4f" % (args.margin, args.miu))
#                 if args.dataset != "vggface2":
#                     if args.len[i] != 36:
#                         feature_dim = 512
#                     else:
#                         feature_dim = 516
#                 else:
#                     feature_dim = num_s * words_s
#                 train(args.save[i], args.len[i], num_s, words_s, feature_dim=feature_dim)




#Cái gì đó nó tăng lên gấp 4 lần 
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# from datetime import datetime
# import torch.distributions as Distributions
# import math
# import argparse
# import sys
# import time
# import os
# from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
# from utils import Logger, AverageMeter, compute_quant, compute_quant_indexing, PqDistRet_Ortho, PqDistRet_Ortho_safe
# from backbone import resnet20_pq, SphereNet20_pq, EdgeFaceBackbone
# from margin_metric import OrthoPQ, CosFace
# from data_loader import get_datasets_transform

# parser = argparse.ArgumentParser(description='PyTorch Implementation of Orthonormal Product Quantization for Scalable Face Image Retrieval')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate mode turned on')
# parser.add_argument('-c', '--cross-dataset', action='store_true', help='generalize on unseen identities')
# parser.add_argument('--bs', type=int, default=256, help='Batch size of each iteration')
# parser.add_argument('--save', nargs='+', help='path to saving models, accept multiple arguments as list')
# parser.add_argument('--load', nargs='+', help='path to loading models, accept multiple arguments as list')
# parser.add_argument('--len', nargs='+', type=int, help='length of hashing codes, accept multiple arguments as list')
# parser.add_argument('--dataset', type=str, default='facescrub', help='which dataset for training (one of facescrub, youtube, CFW, and VGGFace2)')
# parser.add_argument('--num', nargs='+', type=int, help='num. of codebooks, could be 4, 8...')
# parser.add_argument('--words', nargs='+', type=int, default=[256, 256, 256, 256], help='num of words, should be exponential of 2')
# parser.add_argument('--margin', default=0.4, type=float, help='margin of cosine similarity')
# parser.add_argument('--miu', default=0.1, type=float, help='Balance weight of redundancy loss')
# parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'edgeface'], help='Backbone type: resnet or edgeface')
# parser.add_argument('--data_dir', type=str, default='/kaggle/input/facescrub-0210-3', help='Data direction on kaggle for multiple dataset')
# parser.add_argument('--sc', default=30, type=float, help='scale s for initialize metric')
# parser.add_argument('--pretrain_cosface', action='store_true', help='Pretrain with CosFace loss before OrthoPQ')

# try:
#     args = parser.parse_args()
# except Exception as e:
#     print(f"Parser error: {e}")
#     sys.exit(1)

# trainset, testset = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['dataset']
# transform_train, transform_test = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['transform']

# train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=4)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=4)

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# torch.cuda.manual_seed_all(1)

# def train(save_path, length, num, words, feature_dim):
#     best_acc = 0
#     best_mAP = 0
#     best_epoch = 1
#     print('==> Building model..')
#     num_classes = len(trainset.classes)
#     print("number of identities: ", num_classes)
#     print("number of training images: ", len(trainset))
#     print("number of test images: ", len(testset))
#     print("number of training batches per epoch:", len(train_loader))
#     print("number of testing batches per epoch:", len(test_loader))

#     if args.cross_dataset or args.dataset == "vggface2":
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     else:
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)

#     net = nn.DataParallel(net).to(device)
#     cudnn.benchmark = True

#     if args.pretrain_cosface:
#         print("Pre-training with CosFace loss...")
#         metric = CosFace(in_features=feature_dim, out_features=num_classes, s=64.0, m=0.35)
#         metric = nn.DataParallel(metric).to(device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.AdamW([
#             {'params': net.parameters(), 'lr': args.lr * 0.1},  # Backbone: lr=0.0001
#             {'params': metric.parameters(), 'lr': args.lr * 10}  # CosFace head: lr=0.001
#         ], weight_decay=5e-4)
#         scheduler = CosineAnnealingLR(optimizer, T_max=50)
#         checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#         os.makedirs(checkpoint_dir, exist_ok=True)

#         for epoch in range(50):
#             net.train()
#             metric.train()
#             losses = AverageMeter()
#             grad_norm_backbone = 0
#             grad_norm_metric = 0
#             correct = 0
#             total = 0
#             start = time.time()
#             for batch_idx, (inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 transformed_images = transform_train(inputs)
#                 features = net(transformed_images)
#                 outputs = metric(features, targets)
#                 loss = criterion(outputs, targets)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 grad_norm_b = torch.norm(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])).item()
#                 grad_norm_m = torch.norm(torch.cat([p.grad.flatten() for p in metric.parameters() if p.grad is not None])).item()
#                 optimizer.step()
#                 losses.update(loss.item(), len(inputs))
#                 grad_norm_backbone += grad_norm_b
#                 grad_norm_metric += grad_norm_m
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()

#             # In log trung bình mỗi epoch
#             avg_loss = losses.avg
#             avg_grad_norm_backbone = grad_norm_backbone / len(train_loader)
#             avg_grad_norm_metric = grad_norm_metric / len(train_loader)
#             accuracy = 100. * correct / total
#             epoch_elapsed = time.time() - start
#             print(f"Pre-train Epoch {epoch+1} | Loss: {avg_loss:.4f} | Grad_norm_backbone: {avg_grad_norm_backbone:.4f} | Grad_norm_metric: {avg_grad_norm_metric:.4f} | Accuracy: {accuracy:.2f}%")

#             # Đánh giá test accuracy mỗi 5 epoch
#             if (epoch + 1) % 5 == 0:
#                 net.eval()
#                 metric.eval()
#                 test_correct = 0
#                 test_total = 0
#                 with torch.no_grad():
#                     for inputs, targets in test_loader:
#                         inputs, targets = inputs.to(device), targets.to(device)
#                         features = net(transform_test(inputs))
#                         outputs = metric(features, targets)
#                         _, predicted = outputs.max(1)
#                         test_total += targets.size(0)
#                         test_correct += predicted.eq(targets).sum().item()
#                 test_accuracy = 100. * test_correct / test_total
#                 print(f"[Test Phase] Epoch: {epoch+1} | Test Accuracy: {test_accuracy:.2f}%")

#             scheduler.step()

#         print("Saving pre-trained model...")
#         torch.save({'backbone': net.state_dict()}, os.path.join(checkpoint_dir, save_path))
#         return

#     if args.load:
#         checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#         checkpoint_path = os.path.join(checkpoint_dir, args.load[0])
#         checkpoint = torch.load(checkpoint_path)
#         net.load_state_dict(checkpoint['backbone'])
#         print(f"Loaded pretrained weights from {checkpoint_path}")

#     d = int(feature_dim / num)
#     matrix = torch.randn(d, d)
#     for k in range(d):
#         for j in range(d):
#             matrix[j, k] = math.cos((j+0.5)*k*math.pi/d)
#     matrix[:, 0] /= math.sqrt(2)
#     matrix /= math.sqrt(d/2)
#     code_books = torch.Tensor(num, d, words)
#     code_books[0] = matrix[:, :words]
#     for i in range(1, num):
#         code_books[i] = matrix @ code_books[i-1]
#     code_books /= torch.norm(code_books, dim=1, keepdim=True)
#     print("Codebook norms:", [torch.norm(code_books[i], dim=1).mean().item() for i in range(num)])

#     metric = OrthoPQ(in_features=feature_dim, out_features=num_classes, num_books=num, num_words=words, code_books=code_books, sc=args.sc, m=args.margin)
#     metric = nn.DataParallel(metric).to(device)

#     criterion = nn.CrossEntropyLoss()
#     num_books = num
#     num_words = words
#     len_word = int(feature_dim / num_books)
#     len_bit = int(num_books * math.log(num_words, 2))
#     assert length == len_bit, f"Code length mismatch: expected {length}-bit, got {len_bit}-bit"
#     print("num. of codebooks: ", num_books)
#     print("num. of words per book: ", num_words)
#     print("dim. of word: ", len_word)
#     print("code length: %d-bit \t learning rate: %.3f \t scale length: %d \t penalty margin: %.2f \t balance_weight: %.3f" % 
#           (len_bit, args.lr, metric.module.s, metric.module.m, args.miu))

#     optimizer_params = [
#         {'params': metric.parameters(), 'lr': args.lr},
#         {'params': [p for p in net.parameters() if p.requires_grad], 'lr': args.lr * 0.1}
#     ]
#     optimizer = optim.SGD(optimizer_params, weight_decay=1e-3, momentum=0.9)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
#     EPOCHS = 200 if args.dataset in ["facescrub", "cfw", "youtube"] else 160

#     since = time.time()
#     best_loss = 1e3

#     for epoch in range(EPOCHS):
#         print('==> Epoch: %d' % (epoch+1))
#         net.train()
#         metric.train()
#         losses = AverageMeter()
#         loss_clf_avg = AverageMeter()
#         loss_entropy_avg = AverageMeter()
#         grad_norm_backbone = 0
#         grad_norm_metric = 0
#         correct = 0
#         total = 0
#         start = time.time()
#         for batch_idx, (inputs, targets) in enumerate(train_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             transformed_images = transform_train(inputs)
#             features = net(transformed_images)
#             output1, output2, xc_probs = metric(features, targets)
#             loss_clf1 = [criterion(output1[:, i, :], targets) for i in range(num_books)]
#             loss_clf2 = [criterion(output2[:, i, :], targets) for i in range(num_books)]
#             loss_clf = 0.5 * (sum(loss_clf1) / len(loss_clf1) + sum(loss_clf2) / len(loss_clf2))
#             xc_entropy = [Distributions.categorical.Categorical(probs=xc_probs[:, i, :]).entropy().sum() for i in range(num_books)]
#             loss_entropy = sum(xc_entropy) / (num_books * len(inputs))
#             loss = loss_clf + args.miu * loss_entropy
#             optimizer.zero_grad()
#             loss.backward()
#             grad_norm_b = torch.norm(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])).item()
#             grad_norm_m = torch.norm(torch.cat([p.grad.flatten() for p in metric.parameters() if p.grad is not None])).item()
#             optimizer.step()
#             losses.update(loss.item(), len(inputs))
#             loss_clf_avg.update(loss_clf.item(), len(inputs))
#             loss_entropy_avg.update(loss_entropy.item(), len(inputs))
#             grad_norm_backbone += grad_norm_b
#             grad_norm_metric += grad_norm_m
#             _, predicted = output1[:, 0, :].max(1)  # Dùng output1 của codebook đầu để tính accuracy
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#         # In log trung bình mỗi epoch
#         avg_loss = losses.avg
#         avg_loss_clf = loss_clf_avg.avg
#         avg_loss_entropy = loss_entropy_avg.avg
#         avg_grad_norm_backbone = grad_norm_backbone / len(train_loader)
#         avg_grad_norm_metric = grad_norm_metric / len(train_loader)
#         accuracy = 100. * correct / total
#         epoch_elapsed = time.time() - start
#         print(f'Epoch: {epoch+1} | Loss_clf: {avg_loss_clf:.4f} | Loss_entropy: {avg_loss_entropy:.4f} | Total Loss: {avg_loss:.4f} | Grad_norm_backbone: {avg_grad_norm_backbone:.4f} | Grad_norm_metric: {avg_grad_norm_metric:.4f} | Accuracy: {accuracy:.2f}%')
#         print("Epoch Completed in {:.0f}min {:.0f}s".format(epoch_elapsed // 60, epoch_elapsed % 60))
#         scheduler.step(avg_loss)

#         if (epoch+1) % 5 == 0:
#             net.eval()
#             with torch.no_grad():
#                 mlp_weight = metric.module.mlp
#                 index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
#                 queries, test_labels = compute_quant(transform_test, test_loader, net, device)
#                 start = time.time()
#                 mAP, top_k = PqDistRet_Ortho(queries, test_labels, train_labels, index, mlp_weight, len_word, num_books, device, top=50)
#                 time_elapsed = time.time() - start
#                 print("Code generated in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#                 print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 best_mAP = mAP
#                 print('Saving..')
#                 checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#                 os.makedirs(checkpoint_dir, exist_ok=True)
#                 torch.save({'backbone': net.state_dict(), 'mlp': metric.module.mlp}, os.path.join(checkpoint_dir, save_path))
#                 best_epoch = epoch + 1
#     time_elapsed = time.time() - since
#     print("Training Completed in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#     print("Best mAP {:.4f} at epoch {}".format(best_mAP, best_epoch))
#     print("Model saved as %s" % save_path)

# def test(load_path, length, num, words, feature_dim=512):
#     print("===============evaluation on model %s===============" % load_path)
#     num_classes = len(trainset.classes)
#     num_classes_test = len(testset.classes)
#     print("number of train identities: ", num_classes)
#     print("number of test identities: ", num_classes_test)
#     print("number of training images: ", len(trainset))
#     print("number of test images: ", len(testset))
#     print("number of training batches per epoch:", len(train_loader))
#     print("number of testing batches per epoch:", len(test_loader))

#     if args.cross_dataset:
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     else:
#         if args.dataset in ["facescrub", "cfw", "youtube"]:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
#         else:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim)

#     net = nn.DataParallel(net).to(device)

#     checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#     checkpoint_path = os.path.join(checkpoint_dir, load_path)
#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint['backbone'])
#     mlp_weight = checkpoint['mlp']
#     len_word = int(feature_dim / num)
#     net.eval()
#     with torch.no_grad():
#         index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
#         start = datetime.now()
#         query_features, test_labels = compute_quant(transform_test, test_loader, net, device)
#         if args.dataset != "vggface2":
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=5)
#         else:
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=10)

#         time_elapsed = datetime.now() - start
#         print("Query completed in %d ms" % int(time_elapsed.total_seconds() * 1000))
#         print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

# if __name__ == "__main__":
#     save_dir = 'log'
#     if args.evaluate:
#         if not args.load:
#             print("Error: --load is required for evaluation mode")
#             sys.exit(1)
#         if len(args.load) != len(args.num) or len(args.load) != len(args.len) or len(args.load) != len(args.words):
#             print("Warning: Args lengths don't match. Adjusting to shortest length.")
#             min_len = min(len(args.load), len(args.num), len(args.len), len(args.words))
#             args.load = args.load[:min_len]
#             args.num = args.num[:min_len]
#             args.len = args.len[:min_len]
#             args.words = args.words[:min_len]
#         for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
#             if args.cross_dataset:
#                 feature_dim = num_s * words_s
#             else:
#                 if args.dataset != "vggface2":
#                     if args.len[i] != 36:
#                         feature_dim = 512
#                     else:
#                         feature_dim = 516
#                 else:
#                     feature_dim = num_s * words_s
#             test(args.load[i], args.len[i], num_s, words_s, feature_dim=feature_dim)
#     else:
#         if not args.save:
#             print("Error: --save is required for training mode")
#             sys.exit(1)
#         if args.pretrain_cosface:
#             sys.stdout = Logger(os.path.join(save_dir,
#                 'cosface_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
#             print("[Configuration] Pre-training on dataset: %s\n Batch_size: %d\n learning rate: %.6f" %
#                   (args.dataset, args.bs, args.lr))
#             train(args.save[0], None, None, None, feature_dim=512)
#         else:
#             if len(args.save) != len(args.num) or len(args.save) != len(args.len) or len(args.save) != len(args.words):
#                 print("Warning: Args lengths don't match. Adjusting to shortest length.")
#                 min_len = min(len(args.save), len(args.num), len(args.len), len(args.words))
#                 args.save = args.save[:min_len]
#                 args.num = args.num[:min_len]
#                 args.len = args.len[:min_len]
#                 args.words = args.words[:min_len]
#             for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
#                 sys.stdout = Logger(os.path.join(save_dir,
#                     str(args.len[i]) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
#                 print("[Configuration] Training on dataset: %s\n  Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n num_books: %d\n num_words: %d" %
#                       (args.dataset, args.len[i], args.bs, args.lr, num_s, words_s))
#                 print("HyperParams:\nmargin: %.3f\t miu: %.4f" % (args.margin, args.miu))
#                 if args.dataset != "vggface2":
#                     if args.len[i] != 36:
#                         feature_dim = 512
#                     else:
#                         feature_dim = 516
#                 else:
#                     feature_dim = num_s * words_s
#                 train(args.save[i], args.len[i], num_s, words_s, feature_dim=feature_dim)






# Code này in nhiều batch quá, sửa lại
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# from datetime import datetime
# import torch.distributions as Distributions
# import math
# import argparse
# import sys
# import time
# import os
# from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
# from utils import Logger, AverageMeter, compute_quant, compute_quant_indexing, PqDistRet_Ortho, PqDistRet_Ortho_safe
# from backbone import resnet20_pq, SphereNet20_pq, EdgeFaceBackbone
# from margin_metric import OrthoPQ, CosFace
# from data_loader import get_datasets_transform


# parser = argparse.ArgumentParser(description='PyTorch Implementation of Orthonormal Product Quantization for Scalable Face Image Retrieval')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate mode turned on')
# parser.add_argument('-c', '--cross-dataset', action='store_true', help='generalize on unseen identities')
# parser.add_argument('--bs', type=int, default=256, help='Batch size of each iteration')
# parser.add_argument('--save', nargs='+', help='path to saving models, accept multiple arguments as list')
# parser.add_argument('--load', nargs='+', help='path to loading models, accept multiple arguments as list')
# parser.add_argument('--len', nargs='+', type=int, help='length of hashing codes, accept multiple arguments as list')
# parser.add_argument('--dataset', type=str, default='facescrub', help='which dataset for training (one of facescrub, youtube, CFW, and VGGFace2)')
# parser.add_argument('--num', nargs='+', type=int, help='num. of codebooks, could be 4, 8...')
# parser.add_argument('--words', nargs='+', type=int, default=[256, 256, 256, 256], help='num of words, should be exponential of 2')
# parser.add_argument('--margin', default=0.4, type=float, help='margin of cosine similarity')
# parser.add_argument('--miu', default=0.1, type=float, help='Balance weight of redundancy loss')
# parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'edgeface'], help='Backbone type: resnet or edgeface')
# parser.add_argument('--data_dir', type=str, default='/kaggle/input/facescrub-0210-3', help='Data direction on kaggle for multiple dataset')
# parser.add_argument('--sc', default=30, type=float, help='scale s for initialize metric')
# parser.add_argument('--pretrain_cosface', action='store_true', help='Pretrain with CosFace loss before OrthoPQ')

# try:
#     args = parser.parse_args()
# except Exception as e:
#     print(f"Parser error: {e}")
#     sys.exit(1)

# trainset, testset = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['dataset']
# transform_train, transform_test = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['transform']

# train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=4)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=4)

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# torch.cuda.manual_seed_all(1)

# # Giữ nguyên hàm train() và test() từ artifact trước
# def train(save_path, length, num, words, feature_dim):
#     best_acc = 0
#     best_mAP = 0
#     best_epoch = 1
#     print('==> Building model..')
#     num_classes = len(trainset.classes)
#     print("number of identities: ", num_classes)
#     print("number of training images: ", len(trainset))
#     print("number of test images: ", len(testset))
#     print("number of training batches per epoch:", len(train_loader))
#     print("number of testing batches per epoch:", len(test_loader))

#     if args.cross_dataset or args.dataset == "vggface2":
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     else:
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)

#     net = nn.DataParallel(net).to(device)
#     cudnn.benchmark = True

#     if args.pretrain_cosface:
#         print("Pre-training with CosFace loss...")
#         metric = CosFace(in_features=feature_dim, out_features=num_classes, s=64.0, m=0.35)
#         metric = nn.DataParallel(metric).to(device)
#         criterion = nn.CrossEntropyLoss()
#         # optimizer = optim.AdamW([{'params': net.parameters()}, {'params': metric.parameters()}], lr=args.lr, weight_decay=5e-4)
#         # scheduler = CosineAnnealingLR(optimizer, T_max=50)
#         # Trong train() của cosface_opqn_main.py, sửa phần pre-train CosFace:
#         optimizer = optim.AdamW([
#             {'params': net.parameters(), 'lr': args.lr * 0.1},  # Backbone: lr=0.0001
#             {'params': metric.parameters(), 'lr': args.lr * 10}  # CosFace head: lr=0.001
#         ], weight_decay=5e-4)
#         scheduler = CosineAnnealingLR(optimizer, T_max=50)
#         checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#         os.makedirs(checkpoint_dir, exist_ok=True)

#         for epoch in range(50):
#             net.train()
#             losses = AverageMeter()
#             start = time.time()
#             for batch_idx, (inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 transformed_images = transform_train(inputs)
#                 features = net(transformed_images)
#                 outputs = metric(features, targets)
#                 loss = criterion(outputs, targets)
#                 optimizer.zero_grad()
#                 # Trong train(), phần pre-train CosFace, sau loss.backward():
#                 loss.backward()
#                 grad_norm_backbone = torch.norm(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])).item()
#                 grad_norm_metric = torch.norm(torch.cat([p.grad.flatten() for p in metric.parameters() if p.grad is not None])).item()
#                 print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Grad_norm_backbone: {grad_norm_backbone:.4f} | Grad_norm_metric: {grad_norm_metric:.4f}")
#                 optimizer.step()
#                 losses.update(loss.item(), len(inputs))
#                 #print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
#             print(f"Pre-train Epoch {epoch+1} | Loss: {losses.avg:.4f}")
#             scheduler.step()
#         # Trong train(), sau vòng lặp pre-train:
#         net.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, targets in test_loader:
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 features = net(transform_test(inputs))
#                 outputs = metric(features, targets)
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()
#         print(f"Pre-train Test Accuracy: {100. * correct / total:.2f}%")

#         print("Saving pre-trained model...")
#         #torch.save({'backbone': net.state_dict()}, os.path.join(checkpoint_dir, save_path.replace('.tar', '_cosface.tar')))
#         torch.save({'backbone': net.state_dict()}, os.path.join(checkpoint_dir, save_path))
#         return

#     if args.load:
#         checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#         checkpoint_path = os.path.join(checkpoint_dir, args.load[0])
#         checkpoint = torch.load(checkpoint_path)
#         net.load_state_dict(checkpoint['backbone'])
#         print(f"Loaded pretrained weights from {checkpoint_path}")

#     d = int(feature_dim / num)
#     matrix = torch.randn(d, d)
#     for k in range(d):
#         for j in range(d):
#             matrix[j, k] = math.cos((j+0.5)*k*math.pi/d)
#     matrix[:, 0] /= math.sqrt(2)
#     matrix /= math.sqrt(d/2)
#     code_books = torch.Tensor(num, d, words)
#     code_books[0] = matrix[:, :words]
#     for i in range(1, num):
#         code_books[i] = matrix @ code_books[i-1]
#     code_books /= torch.norm(code_books, dim=1, keepdim=True)
#     print("Codebook norms:", [torch.norm(code_books[i], dim=1).mean().item() for i in range(num)])

#     metric = OrthoPQ(in_features=feature_dim, out_features=num_classes, num_books=num, num_words=words, code_books=code_books, sc=args.sc, m=args.margin)
#     metric = nn.DataParallel(metric).to(device)

#     criterion = nn.CrossEntropyLoss()
#     num_books = num
#     num_words = words
#     len_word = int(feature_dim / num_books)
#     len_bit = int(num_books * math.log(num_words, 2))
#     assert length == len_bit, f"Code length mismatch: expected {length}-bit, got {len_bit}-bit"
#     print("num. of codebooks: ", num_books)
#     print("num. of words per book: ", num_words)
#     print("dim. of word: ", len_word)
#     print("code length: %d-bit \t learning rate: %.3f \t scale length: %d \t penalty margin: %.2f \t balance_weight: %.3f" % 
#           (len_bit, args.lr, metric.module.s, metric.module.m, args.miu))

#     optimizer_params = [
#         {'params': metric.parameters(), 'lr': args.lr},
#         {'params': [p for p in net.parameters() if p.requires_grad], 'lr': args.lr * 0.1}
#     ]
#     optimizer = optim.SGD(optimizer_params, weight_decay=1e-3, momentum=0.9)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
#     EPOCHS = 200 if args.dataset in ["facescrub", "cfw", "youtube"] else 160

#     since = time.time()
#     best_loss = 1e3

#     for epoch in range(EPOCHS):
#         print('==> Epoch: %d' % (epoch+1))
#         net.train()
#         losses = AverageMeter()
#         start = time.time()
#         for batch_idx, (inputs, targets) in enumerate(train_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             transformed_images = transform_train(inputs)
#             features = net(transformed_images)
#             output1, output2, xc_probs = metric(features, targets)
#             loss_clf1 = [criterion(output1[:, i, :], targets) for i in range(num_books)]
#             loss_clf2 = [criterion(output2[:, i, :], targets) for i in range(num_books)]
#             loss_clf = 0.5 * (sum(loss_clf1) / len(loss_clf1) + sum(loss_clf2) / len(loss_clf2))
#             xc_entropy = [Distributions.categorical.Categorical(probs=xc_probs[:, i, :]).entropy().sum() for i in range(num_books)]
#             loss_entropy = sum(xc_entropy) / (num_books * len(inputs))
#             loss = loss_clf + args.miu * loss_entropy
#             print(f"Batch {batch_idx}/{len(train_loader)} | Loss_clf: {loss_clf.item():.4f} | Loss_entropy: {loss_entropy.item():.4f} | Total: {loss.item():.4f}")
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             losses.update(loss.item(), len(inputs))

#         epoch_elapsed = time.time() - start
#         print('Epoch %d | Loss: %.4f' % (epoch+1, losses.avg))
#         print("Epoch Completed in {:.0f}min {:.0f}s".format(epoch_elapsed // 60, epoch_elapsed % 60))
#         scheduler.step(losses.avg)

#         if (epoch+1) % 5 == 0:
#             net.eval()
#             with torch.no_grad():
#                 mlp_weight = metric.module.mlp
#                 index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
#                 queries, test_labels = compute_quant(transform_test, test_loader, net, device)
#                 start = time.time()
#                 mAP, top_k = PqDistRet_Ortho(queries, test_labels, train_labels, index, mlp_weight, len_word, num_books, device, top=50)
#                 time_elapsed = time.time() - start
#                 print("Code generated in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#                 print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

#             if losses.avg < best_loss:
#                 best_loss = losses.avg
#                 best_mAP = mAP
#                 print('Saving..')
#                 checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#                 os.makedirs(checkpoint_dir, exist_ok=True)
#                 torch.save({'backbone': net.state_dict(), 'mlp': metric.module.mlp}, os.path.join(checkpoint_dir, save_path))
#                 best_epoch = epoch + 1
#     time_elapsed = time.time() - since
#     print("Training Completed in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#     print("Best mAP {:.4f} at epoch {}".format(best_mAP, best_epoch))
#     print("Model saved as %s" % save_path)

# def test(load_path, length, num, words, feature_dim=512):
#     print("===============evaluation on model %s===============" % load_path)
#     num_classes = len(trainset.classes)
#     num_classes_test = len(testset.classes)
#     print("number of train identities: ", num_classes)
#     print("number of test identities: ", num_classes_test)
#     print("number of training images: ", len(trainset))
#     print("number of test images: ", len(testset))
#     print("number of training batches per epoch:", len(train_loader))
#     print("number of testing batches per epoch:", len(test_loader))

#     if args.cross_dataset:
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     else:
#         if args.dataset in ["facescrub", "cfw", "youtube"]:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
#         else:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim)

#     net = nn.DataParallel(net).to(device)

#     checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#     checkpoint_path = os.path.join(checkpoint_dir, load_path)
#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint['backbone'])
#     mlp_weight = checkpoint['mlp']
#     len_word = int(feature_dim / num)
#     net.eval()
#     with torch.no_grad():
#         index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
#         start = datetime.now()
#         query_features, test_labels = compute_quant(transform_test, test_loader, net, device)
#         if args.dataset != "vggface2":
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=5)
#         else:
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=10)

#         time_elapsed = datetime.now() - start
#         print("Query completed in %d ms" % int(time_elapsed.total_seconds() * 1000))
#         print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

# if __name__ == "__main__":
#     save_dir = 'log'
#     if args.evaluate:
#         if not args.load:
#             print("Error: --load is required for evaluation mode")
#             sys.exit(1)
#         if len(args.load) != len(args.num) or len(args.load) != len(args.len) or len(args.load) != len(args.words):
#             print("Warning: Args lengths don't match. Adjusting to shortest length.")
#             min_len = min(len(args.load), len(args.num), len(args.len), len(args.words))
#             args.load = args.load[:min_len]
#             args.num = args.num[:min_len]
#             args.len = args.len[:min_len]
#             args.words = args.words[:min_len]
#         for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
#             if args.cross_dataset:
#                 feature_dim = num_s * words_s
#             else:
#                 if args.dataset != "vggface2":
#                     if args.len[i] != 36:
#                         feature_dim = 512
#                     else:
#                         feature_dim = 516
#                 else:
#                     feature_dim = num_s * words_s
#             test(args.load[i], args.len[i], num_s, words_s, feature_dim=feature_dim)
#     else:
#         if not args.save:
#             print("Error: --save is required for training mode")
#             sys.exit(1)
#         if args.pretrain_cosface:
#             # Pre-train CosFace không cần num, len, words
#             sys.stdout = Logger(os.path.join(save_dir,
#                 'cosface_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
#             print("[Configuration] Pre-training on dataset: %s\n Batch_size: %d\n learning rate: %.6f" %
#                   (args.dataset, args.bs, args.lr))
#             train(args.save[0], None, None, None, feature_dim=512)  # Gọi train với feature_dim mặc định
#         else:
#             # Train OrthoPQ
#             if len(args.save) != len(args.num) or len(args.save) != len(args.len) or len(args.save) != len(args.words):
#                 print("Warning: Args lengths don't match. Adjusting to shortest length.")
#                 min_len = min(len(args.save), len(args.num), len(args.len), len(args.words))
#                 args.save = args.save[:min_len]
#                 args.num = args.num[:min_len]
#                 args.len = args.len[:min_len]
#                 args.words = args.words[:min_len]
#             for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
#                 sys.stdout = Logger(os.path.join(save_dir,
#                     str(args.len[i]) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
#                 print("[Configuration] Training on dataset: %s\n  Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n num_books: %d\n num_words: %d" %
#                       (args.dataset, args.len[i], args.bs, args.lr, num_s, words_s))
#                 print("HyperParams:\nmargin: %.3f\t miu: %.4f" % (args.margin, args.miu))
#                 if args.dataset != "vggface2":
#                     if args.len[i] != 36:
#                         feature_dim = 512
#                     else:
#                         feature_dim = 516
#                 else:
#                     feature_dim = num_s * words_s
#                 train(args.save[i], args.len[i], num_s, words_s, feature_dim=feature_dim)







# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# from datetime import datetime
# import torch.distributions as Distributions
# import math
# import argparse
# import sys
# import time
# import os
# from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
# from utils import Logger, AverageMeter, compute_quant, compute_quant_indexing, PqDistRet_Ortho, PqDistRet_Ortho_safe
# from backbone import resnet20_pq, SphereNet20_pq, EdgeFaceBackbone
# from margin_metric import OrthoPQ, CosFace  # Thêm CosFace
# from data_loader import get_datasets_transform


# parser = argparse.ArgumentParser(description='PyTorch Implementation of Orthonormal Product Quantization for Scalable Face Image Retrieval')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate mode turned on')
# parser.add_argument('-c', '--cross-dataset', action='store_true', help='generalize on unseen identities')
# parser.add_argument('--bs', type=int, default=256, help='Batch size of each iteration')
# parser.add_argument('--save', nargs='+', help='path to saving models, accept multiple arguments as list')
# parser.add_argument('--load', nargs='+', help='path to loading models, accept multiple arguments as list')
# parser.add_argument('--len', nargs='+', type=int, help='length of hashing codes, accept multiple arguments as list')
# parser.add_argument('--dataset', type=str, default='facescrub', help='which dataset for training (one of facescrub, youtube, CFW, and VGGFace2)')
# parser.add_argument('--num', nargs='+', type=int, help='num. of codebooks, could be 4, 8...')
# parser.add_argument('--words', nargs='+', type=int, default=[256, 256, 256, 256], help='num of words, should be exponential of 2')
# parser.add_argument('--margin', default=0.4, type=float, help='margin of cosine similarity')
# parser.add_argument('--miu', default=0.1, type=float, help='Balance weight of redundancy loss')
# parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'edgeface'], help='Backbone type: resnet or edgeface')
# parser.add_argument('--data_dir', type=str, default='/kaggle/input/facescrub-0210-3', help='Data direction on kaggle for multiple dataset')
# parser.add_argument('--sc', default=30, type=float, help='scale s for initialize metric')
# parser.add_argument('--pretrain_cosface', action='store_true', help='Pretrain with CosFace loss before OrthoPQ')  # Thêm argument

# try:
#     args = parser.parse_args()
# except Exception as e:
#     print(f"Parser error: {e}")
#     sys.exit(1)

# trainset, testset = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['dataset']
# transform_train, transform_test = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone)['transform']

# train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=4)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=4)

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# torch.cuda.manual_seed_all(1)

# def train(save_path, length, num, words, feature_dim):
#     best_acc = 0
#     best_mAP = 0
#     best_epoch = 1
#     print('==> Building model..')
#     num_classes = len(trainset.classes)
#     print("number of identities: ", num_classes)
#     print("number of training images: ", len(trainset))
#     print("number of test images: ", len(testset))
#     print("number of training batches per epoch:", len(train_loader))
#     print("number of testing batches per epoch:", len(test_loader))

#     # Khởi tạo net
#     if args.cross_dataset or args.dataset == "vggface2":
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     else:
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)

#     net = nn.DataParallel(net).to(device)
#     cudnn.benchmark = True

#     # Pre-train với CosFace
#     if args.pretrain_cosface:
#         print("Pre-training with CosFace loss...")
#         metric = CosFace(in_features=feature_dim, out_features=num_classes, s=64.0, m=0.35)
#         metric = nn.DataParallel(metric).to(device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.AdamW([{'params': net.parameters()}, {'params': metric.parameters()}], lr=args.lr, weight_decay=5e-4)
#         scheduler = CosineAnnealingLR(optimizer, T_max=50)  # 50 epochs cho pre-train
#         checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#         os.makedirs(checkpoint_dir, exist_ok=True)

#         for epoch in range(50):  # Pre-train 50 epochs
#             net.train()
#             losses = AverageMeter()
#             start = time.time()
#             for batch_idx, (inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 transformed_images = transform_train(inputs)
#                 features = net(transformed_images)
#                 outputs = metric(features, targets)
#                 loss = criterion(outputs, targets)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 losses.update(loss.item(), len(inputs))
#                 print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
#             print(f"Pre-train Epoch {epoch+1} | Loss: {losses.avg:.4f}")
#             scheduler.step()
#         print("Saving pre-trained model...")
#         torch.save({'backbone': net.state_dict()}, os.path.join(checkpoint_dir, save_path.replace('.tar', '_cosface.tar')))
#         return  # Kết thúc sau pre-train

#     # Load pretrained weights nếu có
#     if args.load:
#         checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#         checkpoint_path = os.path.join(checkpoint_dir, args.load[0])
#         checkpoint = torch.load(checkpoint_path)
#         net.load_state_dict(checkpoint['backbone'])
#         print(f"Loaded pretrained weights from {checkpoint_path}")

#     # Khởi tạo OrthoPQ
#     d = int(feature_dim / num)
#     matrix = torch.randn(d, d)
#     for k in range(d):
#         for j in range(d):
#             matrix[j, k] = math.cos((j+0.5)*k*math.pi/d)
#     matrix[:, 0] /= math.sqrt(2)
#     matrix /= math.sqrt(d/2)
#     code_books = torch.Tensor(num, d, words)
#     code_books[0] = matrix[:, :words]
#     for i in range(1, num):
#         code_books[i] = matrix @ code_books[i-1]
#     code_books /= torch.norm(code_books, dim=1, keepdim=True)
#     print("Codebook norms:", [torch.norm(code_books[i], dim=1).mean().item() for i in range(num)])

#     metric = OrthoPQ(in_features=feature_dim, out_features=num_classes, num_books=num, num_words=words, code_books=code_books, sc=args.sc, m=args.margin)
#     metric = nn.DataParallel(metric).to(device)

#     criterion = nn.CrossEntropyLoss()
#     num_books = num
#     num_words = words
#     len_word = int(feature_dim / num_books)
#     len_bit = int(num_books * math.log(num_words, 2))
#     assert length == len_bit, f"Code length mismatch: expected {length}-bit, got {len_bit}-bit"
#     print("num. of codebooks: ", num_books)
#     print("num. of words per book: ", num_words)
#     print("dim. of word: ", len_word)
#     print("code length: %d-bit \t learning rate: %.3f \t scale length: %d \t penalty margin: %.2f \t balance_weight: %.3f" % 
#           (len_bit, args.lr, metric.module.s, metric.module.m, args.miu))

#     optimizer_params = [
#         {'params': metric.parameters(), 'lr': args.lr},
#         {'params': [p for p in net.parameters() if p.requires_grad], 'lr': args.lr * 0.1}
#     ]
#     optimizer = optim.SGD(optimizer_params, weight_decay=1e-3, momentum=0.9)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
#     EPOCHS = 200 if args.dataset in ["facescrub", "cfw", "youtube"] else 160

#     since = time.time()
#     best_loss = 1e3

#     for epoch in range(EPOCHS):
#         print('==> Epoch: %d' % (epoch+1))
#         net.train()
#         losses = AverageMeter()
#         start = time.time()
#         for batch_idx, (inputs, targets) in enumerate(train_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             transformed_images = transform_train(inputs)
#             features = net(transformed_images)
#             output1, output2, xc_probs = metric(features, targets)
#             loss_clf1 = [criterion(output1[:, i, :], targets) for i in range(num_books)]
#             loss_clf2 = [criterion(output2[:, i, :], targets) for i in range(num_books)]
#             loss_clf = 0.5 * (sum(loss_clf1) / len(loss_clf1) + sum(loss_clf2) / len(loss_clf2))
#             xc_entropy = [Distributions.categorical.Categorical(probs=xc_probs[:, i, :]).entropy().sum() for i in range(num_books)]
#             loss_entropy = sum(xc_entropy) / (num_books * len(inputs))
#             loss = loss_clf + args.miu * loss_entropy
#             print(f"Batch {batch_idx}/{len(train_loader)} | Loss_clf: {loss_clf.item():.4f} | Loss_entropy: {loss_entropy.item():.4f} | Total: {loss.item():.4f}")
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             losses.update(loss.item(), len(inputs))

#         epoch_elapsed = time.time() - start
#         print('Epoch %d | Loss: %.4f' % (epoch+1, losses.avg))
#         print("Epoch Completed in {:.0f}min {:.0f}s".format(epoch_elapsed // 60, epoch_elapsed % 60))
#         scheduler.step(losses.avg)

#         if (epoch+1) % 5 == 0:
#             net.eval()
#             with torch.no_grad():
#                 mlp_weight = metric.module.mlp
#                 index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
#                 queries, test_labels = compute_quant(transform_test, test_loader, net, device)
#                 start = time.time()
#                 mAP, top_k = PqDistRet_Ortho(queries, test_labels, train_labels, index, mlp_weight, len_word, num_books, device, top=50)
#                 time_elapsed = time.time() - start
#                 print("Code generated in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#                 print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

#             if losses.avg < best_loss:
#                 best_loss = losses.avg
#                 best_mAP = mAP
#                 print('Saving..')
#                 checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#                 os.makedirs(checkpoint_dir, exist_ok=True)
#                 torch.save({'backbone': net.state_dict(), 'mlp': metric.module.mlp}, os.path.join(checkpoint_dir, save_path))
#                 best_epoch = epoch + 1
#     time_elapsed = time.time() - since
#     print("Training Completed in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#     print("Best mAP {:.4f} at epoch {}".format(best_mAP, best_epoch))
#     print("Model saved as %s" % save_path)

# def test(load_path, length, num, words, feature_dim=512):
#     print("===============evaluation on model %s===============" % load_path)
#     num_classes = len(trainset.classes)
#     num_classes_test = len(testset.classes)
#     print("number of train identities: ", num_classes)
#     print("number of test identities: ", num_classes_test)
#     print("number of training images: ", len(trainset))
#     print("number of test images: ", len(testset))
#     print("number of training batches per epoch:", len(train_loader))
#     print("number of testing batches per epoch:", len(test_loader))

#     if args.cross_dataset:
#         if args.backbone == 'edgeface':
#             net = EdgeFaceBackbone(feature_dim=feature_dim)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     else:
#         if args.dataset in ["facescrub", "cfw", "youtube"]:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
#         else:
#             if args.backbone == 'edgeface':
#                 net = EdgeFaceBackbone(feature_dim=feature_dim)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim)

#     net = nn.DataParallel(net).to(device)

#     checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#     checkpoint_path = os.path.join(checkpoint_dir, load_path)
#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint['backbone'])
#     mlp_weight = checkpoint['mlp']
#     len_word = int(feature_dim / num)
#     net.eval()
#     with torch.no_grad():
#         index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
#         start = datetime.now()
#         query_features, test_labels = compute_quant(transform_test, test_loader, net, device)
#         if args.dataset != "vggface2":
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=5)
#         else:
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=10)

#         time_elapsed = datetime.now() - start
#         print("Query completed in %d ms" % int(time_elapsed.total_seconds() * 1000))
#         print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

# if __name__ == "__main__":
#     save_dir = 'log'
#     if args.evaluate:
#         if len(args.load) != len(args.num) or len(args.load) != len(args.len) or len(args.load) != len(args.words):
#             print("Warning: Args lengths don't match. Adjusting to shortest length.")
#             min_len = min(len(args.load), len(args.num), len(args.len), len(args.words))
#             args.load = args.load[:min_len]
#             args.num = args.num[:min_len]
#             args.len = args.len[:min_len]
#             args.words = args.words[:min_len]
#         for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
#             if args.cross_dataset:
#                 feature_dim = num_s * words_s
#             else:
#                 if args.dataset != "vggface2":
#                     if args.len[i] != 36:
#                         feature_dim = 512
#                     else:
#                         feature_dim = 516
#                 else:
#                     feature_dim = num_s * words_s
#             test(args.load[i], args.len[i], num_s, words_s, feature_dim=feature_dim)
#     else:
#         if len(args.save) != len(args.num) or len(args.save) != len(args.len) or len(args.save) != len(args.words):
#             print("Warning: Args lengths don't match. Adjusting to shortest length.")
#             min_len = min(len(args.save), len(args.num), len(args.len), len(args.words))
#             args.save = args.save[:min_len]
#             args.num = args.num[:min_len]
#             args.len = args.len[:min_len]
#             args.words = args.words[:min_len]
#         for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
#             sys.stdout = Logger(os.path.join(save_dir,
#                 str(args.len[i]) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
#             print("[Configuration] Training on dataset: %s\n  Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n num_books: %d\n num_words: %d" %
#                   (args.dataset, args.len[i], args.bs, args.lr, num_s, words_s))
#             print("HyperParams:\nmargin: %.3f\t miu: %.4f" % (args.margin, args.miu))
#             if args.dataset != "vggface2":
#                 if args.len[i] != 36:
#                     feature_dim = 512
#                 else:
#                     feature_dim = 516
#                 train(args.save[i], args.len[i], num_s, words_s, feature_dim=feature_dim)
#             else:
#                 feature_dim = num_s * words_s
#                 train(args.save[i], args.len[i], num_s, words_s, feature_dim=feature_dim)
