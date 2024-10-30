import torch
import torch.nn as nn
from transformers import ViTModel
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
import numpy as np
import argparse

import sys
sys.path.append('../')
from TransformerModels_pretrain import ViTModel_custom, ViTForImageClassification
import TransformerConfigs_pretrain as configs
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
    num_labels = 10
elif args.dataset == 'CIFAR100':
    train_loader = GetCIFAR100Training(imgSize=224)
    test_loader = GetCIFAR100Validation(imgSize=224)
    num_labels = 100
elif args.dataset == 'TinyImagenet':
    train_loader, test_loader = load_tiny(shuffle=True, is_train=True)
    num_labels = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")


# Load the configuration of target models
model_arch = 'ViT-16'
# Load the pretrained model
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
# print(vit_model.config)
config = configs.get_b16_config()
model = ViTModel_custom(config=config)

# Load the weights from the pretrained model
model.load_state_dict(vit_model.state_dict())
for param in model.parameters():
    param.requires_grad = False

model = ViTForImageClassification(config, model, num_labels)
model = nn.DataParallel(model).cuda()
N_EPOCHS = 50
LR = 1e-4

# Training loop
optimizer = Adam(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
for epoch in trange(N_EPOCHS, desc="Training"):
    train_loss = 0.0
    correct, total = 0, 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = model(x)

        loss = criterion(y_hat, y)
        train_loss += loss.detach().cpu().item() / len(train_loader)
        correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
        total += len(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    train_loss_list.append(train_loss)
    train_acc_list.append(correct/total)
    
    print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
    print(f"Epoch {epoch + 1}/{N_EPOCHS} acc: {correct/total:.2f}")

filename = "../results/{}/{}/weights.pth".format(model_arch, args.dataset)
torch.save(model.module.state_dict(), filename)