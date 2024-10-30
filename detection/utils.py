import torch
import numpy as np

def vis_from_attn(att_mat, layer_index=-1):
    att_mat = att_mat.to('cpu')
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    residual_att = torch.eye(att_mat.size(-1)) 
    aug_att_mat = att_mat + residual_att 
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1) #[num_layers, N, num_tokens, num_tokens]
    
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    v = joint_attentions[layer_index] 
    att_matrix = v[:,0, 1:].detach().numpy()
    return att_matrix #(N, 196)

# Get the attention vector from the specific transformer layer
def get_attn(dataloader, model, device, layer_index):
    attn_all = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        preds = model(x_batch, output_attentions=True)
        att_mat = [t.unsqueeze(0) for t in preds[-1]]
        att_mat = torch.cat(att_mat, dim=0)
        att_mat = torch.transpose(att_mat, 1, 2).detach().cpu()
        del preds
        att_matrix = vis_from_attn(att_mat, layer_index)
        attn_all.append(att_matrix)
    attn_all = np.vstack(attn_all)
    return attn_all

# Extract the successful adversarial samples
def get_success_adv_index(test_loader, advLoader, model, device):
    correct_index = []
    batch_size = 64
    tracker = 0
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        output = model(x_batch)
        pred = torch.argmax(output, 1).detach().cpu()
        index_temp = np.where(pred.numpy() == y_batch.numpy())[0]
        for i in index_temp:
            correct_index.append(i+batch_size*tracker)
        tracker += 1

    adv_index = []
    batch_size = 64
    tracker = 0
    for x_batch, y_batch in advLoader:
        x_batch = x_batch.to(device)
        output = model(x_batch)
        pred = torch.argmax(output, 1).detach().cpu()
        index_temp = np.where(pred.numpy() != y_batch.numpy())[0]
        for i in index_temp:
            adv_index.append(i+batch_size*tracker)
        tracker += 1

    detect_index = np.intersect1d(np.asarray(correct_index), np.asarray(adv_index))
    return detect_index

def l2_distance(matrix1, matrix2):
    dis = np.linalg.norm(matrix1 - matrix2,axis=1)
    return dis

def get_threshold(error_adv_all, drop_rate):#drop_rate is fpr
    num = int(len(error_adv_all) * drop_rate)
    marks = np.sort(error_adv_all)
    thrs = marks[-num]
    return thrs, marks

# Get the CLS representations from the specific transformer layer
def get_cls(test_loader, model, device, layer_index):
    features = {}
    def get_features(name):
        def hook(_, input, output):
            features[output[0].get_device()] = output[0][:,0].detach().cpu()
        return hook 
    model.module.vit.encoder.layer[layer_index].register_forward_hook(get_features('cls'))

    cls_outputs = []
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        preds = model(x_batch)
        for index in list(features.keys()):
            cls_outputs.append(features[index])
        del preds
    cls_outputs = np.vstack(cls_outputs)
    return cls_outputs
