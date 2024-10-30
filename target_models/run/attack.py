import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os
import argparse
sys.path.append('../')
from WhiteBox import PGDAttack, CWAttack_L2, PatchFool
from utils import register_hook_for_resnet
import TransformerConfigs_pretrain as configs
from TransformerModels_pretrain import ViTModel_custom, ViTForImageClassification

sys.path.append('../../')
from load_data import load_tiny, GetCIFAR100Validation, GetCIFAR10Validation
from Evaluations import test_vit



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

def get_aug():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='TinyImagenet', type=str)
    parser.add_argument('--attack', default='PGD', type=str)
    args = parser.parse_args()
    return args

args = get_aug()
if args.dataset == 'CIFAR10':
    test_loader = GetCIFAR10Validation(imgSize=224, ratio=0.2)
    num_labels = 10
elif args.dataset == 'CIFAR100':
    test_loader = GetCIFAR100Validation(imgSize=224, ratio=0.2)
    num_labels = 100
elif args.dataset == 'TinyImagenet':
    test_loader = load_tiny()
    num_labels = 200

#load the classification model
model_arch = 'ViT-16'
config = configs.get_b16_config()
num_patch = 4 # This is the number of altered patches in the PatchFool attack
patch_size = 16

model = ViTModel_custom(config=config)
model = ViTForImageClassification(config, model, num_labels)
filename = "../results/{}/{}/weights.pth".format(model_arch, args.dataset)
model.load_state_dict(torch.load(filename), strict=False)
model = nn.DataParallel(model).cuda()
model.eval()

adv_filepath = "../results/{}/{}/adv_results/".format(model_arch, args.dataset)
if not os.path.isdir(adv_filepath):
    os.mkdir(adv_filepath)
if args.dataset in ['CIFAR10', 'CIFAR100']:
    if args.attack == 'PGD':
        advLoader, _, _ = PGDAttack(device=device, dataLoader=test_loader, model=model, \
                                                    eps=0.03, num_steps=10, step_size=0.003)

    elif args.attack == 'CW':
        advLoader = CWAttack_L2(device=device, dataLoader=test_loader, model=model, \
                            c=1, kappa=50, max_iter=30, learning_rate=0.01)

    elif args.attack == 'SGM':
        # load source model
        resnet_model = models.resnet18(weights='DEFAULT')
        resnet_model.fc = nn.Linear(512, num_labels)
        sur_filename = {'CIFAR10':'../results/BlackBox/cifar10_resnet18_weights.pth',
                        'CIFAR100':'../results/BlackBox/cifar100_resnet18_weights.pth'}
        resnet_model.load_state_dict(torch.load(sur_filename[args.dataset]), strict=False)
        resnet_model = nn.DataParallel(resnet_model).cuda()
        resnet_model.eval()

        arch = 'resnet18'
        gamma = 0.5
        register_hook_for_resnet(resnet_model, arch=arch, gamma=gamma)

        advLoader, _, _ = PGDAttack(device=device, dataLoader=test_loader, model=resnet_model, \
                                                    eps=0.03, num_steps=10, step_size=0.003)
        
    elif args.attack == 'PatchFool':
        n_tokens = int(224/patch_size)*int(224/patch_size) + 1
        advLoader = PatchFool(dataLoader=test_loader, model=model, patch_size=patch_size, num_patch=num_patch, n_tokens=n_tokens)


elif args.dataset == 'TinyImagenet':
    if args.attack == 'PGD':
        advLoader, _, _ = PGDAttack(device=device, dataLoader=test_loader, model=model, \
                                                    eps=0.06, num_steps=10, step_size=0.006)

    elif args.attack == 'CW':
        advLoader = CWAttack_L2(device=device, dataLoader=test_loader, model=model, \
                            c=1, kappa=50, max_iter=30, learning_rate=0.02)
        
    elif args.attack == 'SGM':
        # load source model
        resnet_model = models.resnet18(weights='DEFAULT')
        resnet_model.fc = nn.Linear(512, num_labels)
        resnet_model.load_state_dict(torch.load('../results/BlackBox/tiny_resnet18_weights.pth'), strict=False)
        resnet_model = nn.DataParallel(resnet_model).cuda()
        resnet_model.eval()

        arch = 'resnet18'
        gamma = 0.5
        register_hook_for_resnet(resnet_model, arch=arch, gamma=gamma)

        advLoader, _, _ = PGDAttack(device=device, dataLoader=test_loader, model=resnet_model, \
                                                    eps=0.06, num_steps=10, step_size=0.006)

    elif args.attack == 'PatchFool':
        n_tokens = int(224/patch_size)*int(224/patch_size) + 1
        advLoader = PatchFool(dataLoader=test_loader, model=model, patch_size=patch_size, num_patch=num_patch, n_tokens=n_tokens)

torch.save(advLoader, adv_filepath + args.attack + '_advLoader.pth')
# Classification accuracy on the adversarial examples
_, adv_acc = test_vit(model=model, test_loader=advLoader, device=device)