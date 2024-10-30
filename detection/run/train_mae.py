import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm, trange
import sys
import argparse
import os
sys.path.append('../')
from ViTMAEModels_pretrain import ViTMAEForPreTraining_custom
import ViTMAEConfigs_pretrain as configs
sys.path.append('../../')
from load_data import load_tiny, GetCIFAR100Training, GetCIFAR100Validation, GetCIFAR10Training, GetCIFAR10Validation


def get_aug():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='TinyImagenet', type=str)
    args = parser.parse_args()
    return args

args = get_aug()
if args.dataset == 'CIFAR10':
    train_loader = GetCIFAR10Training(imgSize=224)
    test_loader = GetCIFAR10Validation(imgSize=224)
    N_EPOCHS = 250
    LR = 1.5e-4

elif args.dataset == 'CIFAR100':
    train_loader = GetCIFAR100Training(imgSize=224)
    test_loader = GetCIFAR100Validation(imgSize=224)
    N_EPOCHS = 250
    LR = 1.5e-4

elif args.dataset == 'TinyImagenet':
    train_loader, test_loader = load_tiny(shuffle=True, is_train=True)
    N_EPOCHS = 500
    LR = 2.5e-4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

ratio = 0.5 
config = configs.ViTMAEConfig(ratio=ratio)
model_custom = ViTMAEForPreTraining_custom(config=config)
model_custom = nn.DataParallel(model_custom).cuda()

## training process

optimizer = Adam(model_custom.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer,  T_max=N_EPOCHS)
train_loss_list = []
test_loss_list = []
lr_list = []
for epoch in trange(N_EPOCHS, desc="Training"):
    train_loss = 0.0
    total = 0
    lr_list.append(optimizer.param_groups[0]["lr"])
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
        x, y = batch
        x, y = x.to(device), y.to(device)
        outputs = model_custom(x)

        target = model_custom.module.patchify(x)
        loss = (outputs[1] - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * outputs[2]).sum() / outputs[2].sum()

        train_loss += loss.detach().cpu().item() / len(train_loader)
        total += len(x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    train_loss_list.append(train_loss)
    print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.5f}")
    scheduler.step()

weights_path = '../results/{}/'.format(args.dataset)
if not os.path.isdir(weights_path):
    os.mkdir(weights_path)
torch.save(model_custom.module.state_dict(), weights_path+'weights.pth')




