# Dilate Loss for crafting UAPs </h1>

Code for our anonymous NIPS submission **Crafting Data-free Universal Adversaries with Dilate Loss** 

## Usage Instructions
### 1) Preparation
  1. This code requires tensorflow version 1.12.
  2. For training and testing of proposed algorithm, download pretrained weights of the models using download_weights.sh in model_wts folder
  2. **Only for testing**, download ILSVRC validation set from [link](http://www.image-net.org/challenges/LSVRC/) and update ilsvrc_test.txt and ilsvrc_test_gt.txt in data folder with path to test images and ground truth labels respectively.
### 2) Training
Run the below command with specific model name to craft adversarial perturbation using dilate loss.

   *python uap_seq_dilate.py --model_name vgg_16|vgg_19|resnet_v1_50|resnet_v1_152|inception_v1|inception_v3*

### 3) Testing
To evaluate the crafted perturbation on the downloaded ILSVRC validation, use the python scrit evaluation.py. Run the below command with specific model name chosen.

   *python evaluation.py --model vgg_16|vgg_19|resnet_v1_50|resnet_v1_152|inception_v1|inception_v3*

## Results
The perturbations crafted using the proposed algorithm acheives below fooling rates in data free scenario.

Network	| Fooling Rate
------- | ------------
VGG16 |	51.35%
VGG19 |	49.58%
Resnet 50 |	60.10%
Resnet 152 |	47.37%
Inception v1 |	52.02%
Inception v3 |	47.18%
