import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import sys
import argparse
sys.path.append('../')
import ViTMAEConfigs_pretrain as configs
from ViTMAEModels_pretrain import ViTMAEForPreTraining_custom
from ViTMAEModels_salient import ViTMAEForPreTraining_salient
from utils import get_attn, get_success_adv_index, l2_distance, get_cls
sys.path.append('../../')
from load_data import load_tiny, GetCIFAR100Validation, GetCIFAR10Validation
import DataManagerPytorch as DMP
sys.path.append('../../target_models/')
from TransformerModels_pretrain import ViTModel_custom, ViTForImageClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def get_aug():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='TinyImagenet', type=str)
    parser.add_argument('--attack', default='PGD', type=str)
    parser.add_argument('--detector', default='Attention', type=str) #Attention or CLS
    parser.add_argument('--ratio', default=0.5) #masking ratio
    args = parser.parse_args()
    return args

args = get_aug()
if args.dataset == 'CIFAR10':
    test_loader = GetCIFAR10Validation(imgSize=224, ratio=0.2)
    num_labels = 10
    layer_index = -1
elif args.dataset == 'CIFAR100':
    test_loader = GetCIFAR100Validation(imgSize=224, ratio=0.2)
    num_labels = 100
    layer_index = -1
elif args.dataset == 'TinyImagenet':
    test_loader = load_tiny()
    num_labels = 200
    layer_index = 1

#load the classification model
model_arch = 'ViT-16'
config = configs.get_b16_config()

model = ViTModel_custom(config=config)
model = ViTForImageClassification(config, model, num_labels)
filename = "../../target_models/results/{}/{}/weights.pth".format(model_arch, args.dataset)
model.load_state_dict(torch.load(filename), strict=False)
model = nn.DataParallel(model).cuda()
model.eval()

# load adversarial examples
adv_filepath = "../../target_models/results/{}/{}/adv_results/".format(model_arch, args.dataset)
advLoader = torch.load(adv_filepath + args.attack + '_advLoader.pth')
advLoader.pin_memory_device = 'cuda'

#load the MAE model
config = configs.ViTMAEConfig(ratio=args.ratio)
if args.attack == 'PatchFool':
    model_mae = ViTMAEForPreTraining_salient(config=config)
else: 
    model_mae = ViTMAEForPreTraining_custom(config=config)

weights_path = '../results/{}/'.format(args.dataset)
model_mae.load_state_dict(torch.load(weights_path + 'weights.pth'), strict=False)
model_mae = nn.DataParallel(model_mae).cuda()


# Extract successful adv samples
detect_index = get_success_adv_index(test_loader, advLoader, model, device)

if args.detector == 'Attention':
    attn_test = get_attn(test_loader, model, device, layer_index)
    attn_adv = get_attn(advLoader, model, device, layer_index)
    sim_test_noise_all, sim_adv_noise_all = [], []

    if args.attack == 'PatchFool':
        n_patches = int(224/16) * int(224/16)
        random_index = np.zeros((len(test_loader.dataset), n_patches))
        for i in range(len(random_index)):
            temp = np.random.choice(n_patches, int(n_patches/2), replace=False)
            random_index[i, temp] = 1
        reLoader_test = DMP.recoverall(model_mae, test_loader, device, salient_index=random_index)
        reLoader_adv = DMP.recoverall(model_mae, advLoader, device, salient_index=random_index)   
        attn_adv_noise = get_attn(reLoader_adv, model, device, layer_index)
        attn_test_noise = get_attn(reLoader_test, model, device, layer_index)
        # calculate distances
        sim_test_noise_all.append(l2_distance(attn_test, attn_test_noise))
        sim_adv_noise_all.append(l2_distance(attn_adv, attn_adv_noise))

    else:
        #reconstruct images
        for random_seed in range(2):
            reLoader_adv = DMP.get_reconstructed_dataset(model_mae, advLoader, device, random_seed)
            reLoader_test = DMP.get_reconstructed_dataset(model_mae, test_loader, device, random_seed)
            attn_adv_noise = get_attn(reLoader_adv, model, device, layer_index)
            attn_test_noise = get_attn(reLoader_test, model, device, layer_index)
            # calculate distances
            sim_test_noise_all.append(l2_distance(attn_test, attn_test_noise))
            sim_adv_noise_all.append(l2_distance(attn_adv, attn_adv_noise))

elif args.detector == 'CLS':
    cls_test = get_cls(test_loader, model, device, layer_index)
    cls_adv = get_cls(advLoader, model, device, layer_index)
    sim_test_noise_all, sim_adv_noise_all = [], []
    if args.attack == 'PatchFool':
        n_patches = int(224/16) * int(224/16)
        random_index = np.zeros((len(test_loader.dataset), n_patches))
        for i in range(len(random_index)):
            temp = np.random.choice(n_patches, int(n_patches/2), replace=False)
            random_index[i, temp] = 1
        reLoader_test = DMP.recoverall(model_mae, test_loader, device, salient_index=random_index)
        reLoader_adv = DMP.recoverall(model_mae, advLoader, device, salient_index=random_index)   
        cls_adv_noise = get_cls(reLoader_adv, model, device, layer_index)
        cls_test_noise = get_cls(reLoader_test, model, device, layer_index)
        sim_test_noise_all.append(l2_distance(cls_test, cls_test_noise))
        sim_adv_noise_all.append(l2_distance(cls_adv, cls_adv_noise))

    else:
        for random_seed in range(2):
            reLoader_adv = DMP.get_reconstructed_dataset(model_mae, advLoader, device, random_seed)
            reLoader_test = DMP.get_reconstructed_dataset(model_mae, test_loader, device, random_seed)
            cls_adv_noise = get_cls(reLoader_adv, model, device, layer_index)
            cls_test_noise = get_cls(reLoader_test, model, device, layer_index)
            sim_test_noise_all.append(l2_distance(cls_test, cls_test_noise))
            sim_adv_noise_all.append(l2_distance(cls_adv, cls_adv_noise))


sim_test_noise_all = np.asarray(sim_test_noise_all)
sim_adv_noise_all = np.asarray(sim_adv_noise_all)
sim_test = np.mean(sim_test_noise_all, axis=0)
sim_adv = np.mean(sim_adv_noise_all, axis=0)

sim_test_correct, sim_adv_correct = sim_test[detect_index], sim_adv[detect_index]
sim_all_correct = np.concatenate((sim_test_correct, sim_adv_correct), axis=0)
true_label_correct = [0]*len(sim_test_correct) + [1]*len(sim_adv_correct) 
true_label_correct = np.asarray(true_label_correct)

auc1 = roc_auc_score(true_label_correct, sim_all_correct)
print('AUC score is', auc1)
