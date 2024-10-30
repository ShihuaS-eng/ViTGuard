import torch 

#Class to help with converting between dataloader and pytorch tensor 
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning 
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data  
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through 
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed 
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

#Get the output shape from the dataloader
def GetOutputShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape

#Get the dataset reconstructed by ViTMAE
def get_reconstructed_dataset(model, data_loader, device, random_seed = None):
    torch.manual_seed(2)
    model.eval()

    batchSize = data_loader.batch_size
    numSamples = len(data_loader.dataset)
    xShape = GetOutputShape(data_loader)
    xRe = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    yClean = yClean.type(torch.LongTensor) 

    advSampleIndex = 0 
    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch, random_seed=random_seed, preset_mask=False)
        y = model.module.unpatchify(outputs[1].detach().cpu()) #outputs[1] is logits 

        # visualize the mask
        mask = outputs[2].detach().cpu() 
        #Repeats this tensor along the specified dimensions.
        mask = mask.unsqueeze(-1).repeat(1, 1, model.module.config.patch_size**2 *3)  # (N, H*W, p*p*3) 
        mask = model.module.unpatchify(mask)  # 1 is removing, 0 is keeping. 
        # MAE reconstruction pasted with visible patches
        im_paste = x_batch.detach().cpu() * (1 - mask) + y * mask
        im_paste = torch.clamp(im_paste, 0, 1)
        for j in range(0, batchSize):
            xRe[advSampleIndex] = im_paste[j]
            yClean[advSampleIndex] = y_batch[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
            if advSampleIndex == numSamples:
                break
        del outputs, y, mask, im_paste
    reLoader_adv = TensorToDataLoader(xRe, yClean, transforms= None, batchSize= batchSize, randomizer=None)
    return reLoader_adv

#Apply ViTMAE twice and get the reconstructed dataset
def recoverall(model, data_loader, device, salient_index, random_seed = None):
    torch.manual_seed(2)
    model.eval()

    batchSize = data_loader.batch_size
    numSamples = len(data_loader.dataset)
    xShape = GetOutputShape(data_loader)
    xRe = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    yClean = yClean.type(torch.LongTensor) 
    tracker = 0

    advSampleIndex = 0 
    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        batch_size = len(x_batch)
        salient_index_batch = (1-salient_index[tracker:tracker+batch_size]) #In MAE, 1 is removing, 0 is keeping.
        outputs = model(x_batch, random_seed=random_seed, preset_mask=salient_index_batch)
        y = model.module.unpatchify(outputs[1].detach().cpu())
        mask = outputs[2].detach().cpu() 
        #Repeats this tensor along the specified dimensions.
        mask = mask.unsqueeze(-1).repeat(1, 1, model.module.config.patch_size**2 *3)  
        mask = model.module.unpatchify(mask)  
        im_paste = y * mask
        im_paste = torch.clamp(im_paste, 0, 1)
        del outputs, y, mask

        salient_index_batch = salient_index[tracker:tracker+batch_size] 
        tracker += batch_size  
        outputs = model(x_batch, random_seed=random_seed, preset_mask=salient_index_batch)
        y = model.module.unpatchify(outputs[1].detach().cpu()) #outputs[1] is logits 
        mask = outputs[2].detach().cpu() 
        #Repeats this tensor along the specified dimensions.
        mask = mask.unsqueeze(-1).repeat(1, 1, model.module.config.patch_size**2 *3) 
        mask = model.module.unpatchify(mask) 
        im_paste_2 = im_paste + y * mask
        im_paste_2 = torch.clamp(im_paste_2, 0, 1)
        for j in range(0, batch_size):
            xRe[advSampleIndex] = im_paste_2[j]
            yClean[advSampleIndex] = y_batch[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
            if advSampleIndex == numSamples:
                break
        del outputs, y, mask, im_paste_2
    
    reLoader_adv = TensorToDataLoader(xRe, yClean, transforms= None, batchSize= batchSize, randomizer=None)
    return reLoader_adv





