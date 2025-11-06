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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import Logger, AverageMeter, compute_quant, compute_quant_indexing, PqDistRet_Ortho
from backbone import resnet20_pq, EdgeFaceBackbone
from margin_metric import OrthoPQ
from data_loader import get_datasets_transform

# ============================== ARGUMENTS ==============================
parser = argparse.ArgumentParser(description='PyTorch Implementation of Orthonormal Product Quantization (OPQN-v2.1)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate mode')
parser.add_argument('-c', '--cross-dataset', action='store_true', help='generalize on unseen identities')
parser.add_argument('--bs', type=int, default=256, help='batch size')
parser.add_argument('--save', nargs='+', help='path to saving models')
parser.add_argument('--load', nargs='+', help='path to loading models')
parser.add_argument('--len', nargs='+', type=int, help='code length in bits')
parser.add_argument('--dataset', type=str, default='facescrub', help='dataset: facescrub, youtube, cfw, vggface2')
parser.add_argument('--num', nargs='+', type=int, help='num. of codebooks (4, 8, ...)')
parser.add_argument('--words', nargs='+', type=int, default=[256, 256, 256, 256], help='num of words per book (2**n)')
parser.add_argument('--margin', default=0.4, type=float, help='cosine margin')
parser.add_argument('--miu', default=0.1, type=float, help='entropy loss weight')
parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'edgeface'])
parser.add_argument('--data_dir', type=str, default='/kaggle/input/facescrub-0210-3', help='dataset root')
parser.add_argument('--sc', default=40, type=float, help='scale s for metric init (paper: 40)')  # sửa về 40
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--scheduler_type', default='step', choices=['step', 'plateau'], help='lr scheduler')
parser.add_argument('--freeze', action='store_true', help='freeze backbone (for finetuning)')
try:
    args = parser.parse_args()
except Exception as e:
    print(f"Parser error: {e}")
    sys.exit(1)

# ============================== DATA LOADER ==============================
trainset, testset = get_datasets_transform(
    args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone
)['dataset']
transform_train, transform_test = get_datasets_transform(
    args.dataset, args.data_dir, cross_eval=args.cross_dataset, backbone=args.backbone
)['transform']

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=4)
test_loader  = torch.utils.data.DataLoader(testset,  batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=4)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.cuda.manual_seed_all(1)

# ============================== LR SCHEDULER ==============================
class adjust_lr:
    def __init__(self, step, decay):
        self.step = step
        self.decay = decay
    def adjust(self, optimizer, epoch):
        lr = args.lr * (self.decay ** (epoch // self.step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# ============================== CODEBOOK GENERATOR (CHUẨN ORTHONORMAL) ==============================
def generate_orthonormal_codebooks(num_books, d, words):
    matrix = torch.zeros(d, d)
    for k in range(d):
        for j in range(d):
            matrix[j, k] = math.cos((j + 0.5) * k * math.pi / d)
    matrix[:, 0] /= math.sqrt(2)
    matrix /= math.sqrt(d / 2)                     # chuẩn hóa orthonormal đúng công thức DCT
    code_books = torch.zeros(num_books, d, words)
    code_books[0] = matrix[:, :words]
    for i in range(1, num_books):
        code_books[i] = matrix @ code_books[i-1]   # giữ nguyên tính orthonormal
    # KHÔNG normalize lại → giữ chuẩn toán học!
    print("Codebooks generated (orthonormal). Norms ≈ 1.0")
    print("Norms:", [torch.norm(code_books[i], dim=0).mean().item() for i in range(num_books)])
    return code_books

# ============================== TRAIN ==============================
def train(save_path, length, num, words, feature_dim):
    print('==> Building model..')
    num_classes = len(trainset.classes)
    d = int(feature_dim / num)
    code_books = generate_orthonormal_codebooks(num, d, words)

    # Backbone
    if args.backbone == 'edgeface':
        net = EdgeFaceBackbone(feature_dim=feature_dim)
    else:
        if args.cross_dataset or args.dataset == "vggface2":
            net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
        else:
            net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)

    # Metric
    metric = OrthoPQ(
        in_features=feature_dim,
        out_features=num_classes,
        num_books=num,
        code_books=code_books,
        num_words=words,
        sc=args.sc,
        m=args.margin
    )

    net = nn.DataParallel(net).to(device)
    if args.backbone == 'edgeface' and args.freeze:
        print("Freezing EdgeFace backbone")
        for p in net.module.backbone.parameters():
            p.requires_grad = False
    metric = nn.DataParallel(metric).to(device)
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    len_word = d
    len_bit = int(num * math.log(words, 2))
    assert length == len_bit, f"Code length mismatch: {length} vs {len_bit}"

    print(f"Code: {num} books × {words} words → {len_bit}-bit | sc={metric.module.s} | m={metric.module.m}")

    # Optimizer
    optimizer_params = [{'params': metric.parameters(), 'lr': args.lr}]
    trainable_backbone = [p for p in net.parameters() if p.requires_grad]
    if trainable_backbone:
        optimizer_params.append({'params': trainable_backbone, 'lr': args.lr})
    optimizer = optim.SGD(optimizer_params, weight_decay=args.wd, momentum=0.9)

    EPOCHS = 200 if args.dataset in ["facescrub", "cfw", "youtube"] else 160

    if args.scheduler_type == 'step':
        scheduler = adjust_lr(35 if args.dataset in ["facescrub", "cfw", "youtube"] else 20, 0.5)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_loss = 1e3
    best_mAP = 0
    best_epoch = 1
    since = time.time()

    for epoch in range(EPOCHS):
        print(f'\n==> Epoch: {epoch+1}')
        net.train()
        losses = AverageMeter()
        if args.scheduler_type == 'step':
            scheduler.adjust(optimizer, epoch)

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            x = transform_train(inputs)
            feats = net(x)
            out1, out2, xc_probs = metric(feats, targets)

            loss_clf1 = sum(criterion(out1[:, i, :], targets) for i in range(num)) / num
            loss_clf2 = sum(criterion(out2[:, i, :], targets) for i in range(num)) / num
            loss_clf = 0.5 * (loss_clf1 + loss_clf2)

            loss_entropy = sum(
                Distributions.Categorical(probs=xc_probs[:, i, :]).entropy().sum()
                for i in range(num)
            ) / (num * inputs.size(0))

            loss = loss_clf + args.miu * loss_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), inputs.size(0))

        if args.scheduler_type == 'plateau':
            scheduler.step(losses.avg)

        print(f'Epoch {epoch+1} | Loss: {losses.avg:.4f}')

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            net.eval()
            with torch.no_grad():
                mlp_weight = metric.module.mlp
                index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
                queries, test_labels = compute_quant(transform_test, test_loader, net, device)
                start = time.time()
                mAP, top_k = PqDistRet_Ortho(
                    queries, test_labels, train_labels, index,
                    mlp_weight, len_word, num, device, top=50
                )
                elapsed = time.time() - start
                print(f"Eval time: {elapsed//60:.0f}min {elapsed%60:.0f}s")
                print(f'[Eval] mAP@50: {100*mAP:.2f}% | top-50: {100*top_k:.2f}%')

            if losses.avg < best_loss:
                best_loss = losses.avg
                best_mAP = mAP
                best_epoch = epoch + 1
                print('Saving best model...')
                ckpt_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    'backbone': net.state_dict(),
                    'mlp': metric.module.mlp
                }, os.path.join(ckpt_dir, save_path))

    total_time = time.time() - since
    print(f"\nTraining completed in {total_time//60:.0f}min {total_time%60:.0f}s")
    print(f"Best mAP@50: {100*best_mAP:.2f}% at epoch {best_epoch}")
    print(f"Model saved: {save_path}")

# ============================== TEST ==============================
def test(load_path, length, num, words, feature_dim=512):
    len_bit = int(num * math.log(words, 2))
    assert length == len_bit, f"Code length mismatch: expected {length}, got {len_bit}-bit"

    print(f"\n{'='*60}")
    print(f" EVALUATION ON MODEL: {load_path}")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset} | Code: {num}×{words} → {len_bit}-bit | Backbone: {args.backbone}")
    if args.cross_dataset:
        print("Mode: Cross-dataset evaluation (unseen identities)")

    # In thông tin dataset
    print(f"\nDataset Statistics:")
    print(f"  Train identities: {len(trainset.classes):,} | Images: {len(trainset):,}")
    print(f"  Test  identities: {len(testset.classes):,} | Images: {len(testset):,}")
    print(f"  Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # Tạo codebook orthonormal (giống train)
    d = int(feature_dim / num)
    code_books = generate_orthonormal_codebooks(num, d, words)

    # Khởi tạo backbone
    if args.backbone == 'edgeface':
        net = EdgeFaceBackbone(feature_dim=feature_dim)
    else:
        if args.cross_dataset or args.dataset == "vggface2":
            net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
        else:
            net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)

    net = nn.DataParallel(net).to(device)

    # Xử lý đường dẫn checkpoint (hỗ trợ tuyệt đối + tương đối)
    if os.path.isabs(load_path):
        checkpoint_path = load_path
    else:
        checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
        checkpoint_path = os.path.join(checkpoint_dir, load_path)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"\nLoading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['backbone'])
    mlp_weight = checkpoint.get('mlp')
    if mlp_weight is None:
        print("Warning: 'mlp' key not found in checkpoint. Using default MLP weights.")
    
    len_word = d
    net.eval()
    print(f"Model loaded successfully. Feature dim: {feature_dim}, len_word: {len_word}")

    # Bắt đầu đo thời gian
    total_start = time.perf_counter()

    with torch.no_grad():
        print(f"\nComputing train index ({len(trainset):,} images)...")
        start = time.perf_counter()
        index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
        index_time = time.perf_counter() - start
        print(f"Train index built in {index_time:.2f}s")

        print(f"Extracting test features ({len(testset):,} images)...")
        start = time.perf_counter()
        query_features, test_labels = compute_quant(transform_test, test_loader, net, device)
        feat_time = time.perf_counter() - start
        print(f"Test features extracted in {feat_time:.2f}s")

        # mAP@all (chuẩn paper)
        print(f"\nComputing mAP@all (top-{len(trainset)})...")
        start = time.perf_counter()
        mAP, _ = PqDistRet_Ortho(
            query_features, test_labels, train_labels, index,
            mlp_weight, len_word, num, device, top=len(trainset)
        )
        map_time = time.perf_counter() - start
        map_time_ms = map_time * 1000
        map_per_query = map_time_ms / len(testset)

        print(f"\nFINAL RESULTS:")
        print(f"  mAP@all          : {100 * mAP:.4f}%")
        print(f"  mAP time         : {map_time_ms:.2f} ms total | {map_per_query:.4f} ms/query")

        # Top-k từ 1 đến 100
        print(f"\nTop-k Accuracy:")
        topk_values = [1, 5, 10, 20, 50, 100]
        for k in topk_values:
            if k > len(trainset):
                break
            _, top_k = PqDistRet_Ortho(
                query_features, test_labels, train_labels, index,
                mlp_weight, len_word, num, device, top=k
            )
            print(f"  top-{k:3d}          : {100 * top_k:.4f}%")

        # Tổng thời gian
        total_time = time.perf_counter() - total_start
        total_ms = total_time * 1000
        avg_ms = total_ms / len(testset)

        print(f"\nPerformance Summary:")
        print(f"  Total query time     : {total_ms:.2f} ms")
        print(f"  Average per query    : {avg_ms:.4f} ms/query")
        print(f"  QPS (queries/sec)    : {1000/avg_ms:.2f}")
        print(f"{'='*60}")

# ============================== MAIN ==============================
if __name__ == "__main__":
    save_dir = 'log'
    os.makedirs(save_dir, exist_ok=True)

    # Sync list lengths
    def sync_lists(*lists):
        min_len = min(len(l) for l in lists)
        return [l[:min_len] for l in lists]

    if args.evaluate:
        args.load, args.num, args.len, args.words = sync_lists(args.load, args.num, args.len, args.words)
        for i, (n, w) in enumerate(zip(args.num, args.words)):
            feature_dim = n * w if (args.cross_dataset or args.dataset == "vggface2") else (516 if args.len[i] == 36 else 512)
            test(args.load[i], args.len[i], n, w, feature_dim)
    else:
        args.save, args.num, args.len, args.words = sync_lists(args.save, args.num, args.len, args.words)
        for i, (n, w) in enumerate(zip(args.num, args.words)):
            log_file = os.path.join(save_dir,
                f"{args.len[i]}bits_{args.dataset}_{datetime.now().strftime('%m%d%H%M')}.txt")
            sys.stdout = Logger(log_file)

            print("[Config]", args.dataset, f"{args.len[i]}-bit", f"bs={args.bs}", f"lr={args.lr}",
                  f"books={n}", f"words={w}", f"backbone={args.backbone}")
            print("Hyperparams: margin=%.3f  miu=%.3f  sc=%.1f" % (args.margin, args.miu, args.sc))

            feature_dim = n * w if args.dataset == "vggface2" else (516 if args.len[i] == 36 else 512)
            train(args.save[i], args.len[i], n, w, feature_dim)