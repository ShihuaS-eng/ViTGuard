# ViTGuard (under Artifact Evaluation from ACSAC 2024)
Code for the paper, "ViTGuard: Attention-aware Detection against Adversarial Examples for Vision Transformer".

We provide the code for detecting four representative adversarial attacks: PGD (gradient-based), CW (optimization-based), Patch-Fool (attention-aware), and SGM (transfer-based). 

## Software Installation
We use the following software packages: 
<ul>
  <li>python==3.10.12</li>
  <li>pytorch==2.0.1</li>
  <li>torchvision==0.15.2</li>
  <li>numpy==1.26.0</li>
  <li>transformers==4.33.3</li>
</ul>

Run `pip3 install -r requirements.txt` to install the required packages.

## Code Structure
The source code is structured into two main subfolders: `target_models` and `detection`. Within the target_models folder, you will find configurations for target ViT classifiers and adversarial attacks. The detection folder includes settings for the MAE model used in image reconstruction, along with configurations for ViTGuard detectors.

## Running the Code
**Note:** Step 1 and Step 2 are optional, as the weights of the target model for the TinyImagenet dataset are available for download from [this link](https://drive.google.com/file/d/1zwqrSesNPtaQTeTbfv7rMon9IW2i0Sof/view?usp=sharing). After downloading the file, move it to the `target_models` directory and unzip it. Additionally, the `results` folder also contains the adversarial examples generated using the TinyImagenet dataset.

**Note:** Step 3(1) is optional, as the model weights for ViTMAE are available for download from [this link](https://drive.google.com/file/d/13KE103qawMLhIeBE8-U4kr9GoagkOwCs/view?usp=sharing). After downloading the file, move it to the `detection` directory and unzip it.

**Note:** Users can proceed directly to Step 3(2) to execute the detection process.

### Step 1. Train a target model
In the main directory, run `cd target_models/run`

A target ViT model can be trained by running

    python3 train.py --dataset TinyImagenet

The model will be trained, saved into the `target_models/results/ViT-16/TinyImagenet/` subfolder, and named
`weights.pth`. The dataset used for training can be modified to `CIFAR10` or `CIFAR100` as needed.

### Step 2. Craft adversarial samples
To craft adversarial samples, run

    python3 attack.py --dataset TinyImagenet --attack PGD
    
The DataLoader holding the adversarial samples will be stored in the `target_models/results/ViT-16/TinyImagenet/adv_results` subfolder.

In this example, the PGD attack is utilized; however, it can be substituted with other attack methods, such as `CW`, `SGM`, or `PatchFool`. The dataset can be changed to `CIFAR10` or `CIFAR100`. The `target_models/run/Table1.ipynb` shows the classification accuracy of adversarial examples generated by various attacks.


### Step 3. Detect adversarial samples
The detection mechanism comprises two stages: (1) training an MAE model for image reconstruction and (2) employing ViTGuard detectors. 

In the main directory, run `cd detection/run`

(1) To train an MAE model, run

    python3 train_mae.py --dataset TinyImagenet
    
The model will be trained, saved into the `detection/results/TinyImagenet/` subfolder, and named `weights.pth`.

(2) We proposed two individual detectors based on the attention and CLS representation, respectively. To get the AUC score for the detection method, run

    python3 detection.py --dataset TinyImagenet --attack PGD --detector Attention

The detector can also be replaced with `CLS` to evaluate the CLS-based detector.

