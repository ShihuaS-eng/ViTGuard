import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append('/home/artifacts/ViTGuard-main/')
import DataManagerPytorch as DMP

sys.path.append('target_models/')
from utils import clamp, PCGrad

def PGDAttack(device, dataLoader, model, eps, num_steps, step_size, clipMin=0, clipMax=1):
    model.eval() 
    numSamples = len(dataLoader.dataset)
    xShape = DMP.GetOutputShape(dataLoader)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    #Go through each sample 
    tracker = 0

    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        #Put the data from the batch onto the device 
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        xData = xData.to(device)
        yData = yData.type(torch.LongTensor).to(device)

        # Forward pass the data through the model
        # Calculate the loss
        loss = torch.nn.CrossEntropyLoss()

        for i in range(num_steps):
            # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
            xDataTemp.requires_grad = True

            output = model(xDataTemp)

            # Zero all existing gradients
            model.zero_grad()
            # Calculate gradients of model in backward pass
            cost = loss(output, yData)
            cost.backward()
            # Collect the element-wise sign of the data gradient
            signDataGrad = xDataTemp.grad.data.sign()
            xDataTemp = xDataTemp.detach() + step_size*signDataGrad #FGSM
            # perturbedImage = perturbedImage.detach().cpu()

            # Adding clipping to maintain the range
            delta = torch.clamp(xDataTemp-xData, min=-eps, max=eps)
            xDataTemp = torch.clamp(xData+delta, clipMin, clipMax)

        # xDataTemp = xDataTemp.detach().cpu()
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xDataTemp[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
        del xDataTemp
        del signDataGrad
        torch.cuda.empty_cache()    
    #All samples processed, now time to save in a dataloader and return
    yClean = yClean.type(torch.LongTensor)
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader, xAdv.numpy(), yClean.numpy()




class CW:
    def __init__(self, model, device=None, c=1, kappa=0, steps=50, lr=0.01):
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.device = device
        self.model = model
    
    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)
    
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()
        
        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            # outputs = self.get_logits(adv_images)
            outputs = self.model(adv_images)
            f_loss = self.f(outputs, labels).sum()
            cost = L2_loss + self.c*f_loss
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        best_adv_images = self.tanh_space(w).detach()
        return best_adv_images


    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x*2-1, min=-1, max=1))

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # find the max logit other than the target class
        other = torch.max((1-one_hot_labels)*outputs, dim=1)[0]
        # get the target class's logit
        real = torch.max(one_hot_labels*outputs, dim=1)[0]
        
        return torch.clamp((real-other), min=-self.kappa)


def CWAttack_L2(device, dataLoader, model, c=1e-4, kappa=0, max_iter=30, learning_rate=0.001):

    model.eval() 
    numSamples = len(dataLoader.dataset)
    xShape = DMP.GetOutputShape(dataLoader)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 

    cw_attack = CW(model=model, device=device, c=c, kappa=kappa, steps=max_iter, lr=learning_rate)

    for images, labels in dataLoader:
        batchSize = images.shape[0]
        adversarial_images = cw_attack.forward(images, labels)

        for j in range(0, batchSize):
            xAdv[advSampleIndex] = adversarial_images[j]
            yClean[advSampleIndex] = labels[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index

        del adversarial_images
        torch.cuda.empty_cache()   

    #All samples processed, now time to save in a dataloader and return
    yClean = yClean.type(torch.LongTensor)
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

def PatchFool(dataLoader, model, atten_select=5, patch_size=16, num_patch=4, n_tokens=197):
    '''
    atten_select: Select patch based on which attention layer
    num_patch: the number of patches selected
    '''
    criterion = nn.CrossEntropyLoss().cuda()
    attack_learning_rate = 0.05
    step_size = 1
    gamma = 0.95
    train_attack_iters = 250
    atten_loss_weight = 0.002

    mu = [0, 0, 0]
    std = [1,1,1]
    mu = torch.tensor(mu).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()


    numSamples = len(dataLoader.dataset)
    xShape = DMP.GetOutputShape(dataLoader)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    tracker = 0

    for i, (X, y) in enumerate(dataLoader):
        batchSize = X.shape[0]
        tracker = tracker + batchSize
    
        X, y = X.cuda(), y.cuda()
        patch_num_per_line = int(X.size(-1) / patch_size)
        delta = torch.zeros_like(X).cuda()
        delta.requires_grad = True

        model.zero_grad()
        out, atten = model(X + delta, output_attentions=True) 
        loss = criterion(out, y)

        ### choose patch
        atten_layer = atten[atten_select].mean(dim=1) #average across heads
        atten_layer = atten_layer.mean(dim=-2)[:, 1:] 
        max_patch_index = atten_layer.argsort(descending=True)[:, :num_patch]

        #build mask
        mask = torch.zeros([X.size(0), 1, X.size(2), X.size(3)]).cuda()
        for j in range(X.size(0)):
            index_list = max_patch_index[j]
            for index in index_list:
                row = (index // patch_num_per_line) * patch_size
                column = (index % patch_num_per_line) * patch_size
                mask[j, :, row:row + patch_size, column:column + patch_size] = 1

        # adv attack
        max_patch_index_matrix = max_patch_index[:, 0] 
        max_patch_index_matrix = max_patch_index_matrix.repeat(n_tokens, 1) 
        max_patch_index_matrix = max_patch_index_matrix.permute(1, 0) 
        max_patch_index_matrix = max_patch_index_matrix.flatten().long() 

        delta = torch.rand_like(X)
        X = torch.mul(X, 1 - mask)

        delta = delta.cuda()
        delta.requires_grad = True
        opt = torch.optim.Adam([delta], lr=attack_learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

        # start adv attack
        for train_iter_num in range(train_attack_iters):
            model.zero_grad()
            opt.zero_grad()
            out, atten = model(X + torch.mul(delta, mask), output_attentions=True)

            '''final CE-loss'''
            loss = criterion(out, y)
            grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
            ce_loss_grad_temp = grad.view(X.size(0), -1).detach().clone()
            # Attack the first 6 layers' Attn
            range_list = range(len(atten)//2)
            for atten_num in range_list:
                if atten_num == 0:
                    continue
                atten_map = atten[atten_num] 
                atten_map = atten_map.mean(dim=1) 
                atten_map = atten_map.view(-1, atten_map.size(-1)) 
                atten_map = -torch.log(atten_map) 
                atten_loss = F.nll_loss(atten_map, max_patch_index_matrix)
                atten_grad = torch.autograd.grad(atten_loss, delta, retain_graph=True)[0]
                atten_grad_temp = atten_grad.view(X.size(0), -1)
                cos_sim = F.cosine_similarity(atten_grad_temp, ce_loss_grad_temp, dim=1)
                atten_grad = PCGrad(atten_grad_temp, ce_loss_grad_temp, cos_sim, grad.shape)
                grad += atten_grad * atten_loss_weight


            opt.zero_grad()
            delta.grad = -grad
            opt.step()
            scheduler.step()
        delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)
        # Eval Adv Attack
        with torch.no_grad():
            perturb_x = X + torch.mul(delta, mask)
            out = model(perturb_x)
            classification_result_after_attack = out.max(1)[1] == y
            loss = criterion(out, y)
            print(classification_result_after_attack.sum().item())
    
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = perturb_x[j]
            yClean[advSampleIndex] = y[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
        yClean = yClean.type(torch.LongTensor)
        advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader