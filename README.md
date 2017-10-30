# GANCS
Compressed Sensing MRI based on Generative Adversarial Network 

# Introduction

## First Authors:
Morteza Mardani, Enhao Gong

## Arxiv Paper
Deep Generative Adversarial Networks for Compressed Sensing Automates MRI
Arxiv Paper: https://arxiv.org/abs/1706.00051

## References:
The basic code base is derived from super resolution github repo. https://github.com/david-gpu/srez

https://github.com/david-gpu/srez
https://arxiv.org/abs/1609.04802
https://github.com/carpedm20/DCGAN-tensorflow
http://wiseodd.github.io/techblog/2017/03/02/least-squares-gan/
http://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/


# Method

## Deep Generative Adversarial Networks for CS MRI
Magnetic resonance imaging (MRI) suffers from aliasing artifacts when it is highly undersampled for fast imaging. Conventional CS MRI reconstruction uses regularized iterative reconstruction based on pre-defined sparsity transform, which usually include time-consuming iterative optimization and may result in undesired artifacts such as oversmoothing. Here we propose a novel CS framework that permeates benefits from deep learning and generative adversarial networks (GAN) to modeling a manifold of MR images from historical patients. Extensive evaluations on a large MRI datasets of pediatric pateints show it results in superior perforamnce, retrieves image with improved quality and finer details relative to conventional CS and pixel-wise deep learning schemes. 

## descriminator related loss
We have been exploring different loss functions for GAN, including:

* log-loss
* LS loss (better than log-loss, use as default, easy to tune and optimize)
* Cycle-GAN/WGAN loss (todo)

## Undersampling
### 1D undersampling
1D undersampling is generated using the variable density distribution. 
`R_factor` defines the desired reduction factor, which controls how many samples to randomly pick. 
`R_alpha` defines the decay of VD distribution with formula `p=x^alpha`.
`R_seed` defines the random seed used, for negative values there is no fixed undersampling pattern.

### 1D/2D undersampling
`sampling_pattern` can be a path to .mat file for specific 1D/2D undersampling mask


## Generator Model
Several models are explored including ResNet-ish models from super-resolution paper and encoder-decoder models.

## Descriminator Model
Currently we are using 4*(Conv-BN-RELU-POOL)+2*(CONV-BN-RELU)+CONV+MEAN+softmax (for logloss)

## Loss formulation
Loss is a mixed combination with: 1) Data consistency loss, 2) pixel-wise MSE/L1/L2 loss and 3) LS-GAN loss

`FLAGS.gene_log_factor = 0 # log loss vs least-square loss`

`FLAGS.gene_dc_factor = 0.9 # data-consistency (kspace) loss vs generator loss`

`FLAGS.gene_mse_factor = 0.001 # simple MSE loss originally for forward-passing model vs GAN loss`
GAN loss = generator loss + discriminator loss

`gene_fool_loss = FLAGS.gene_log_factor * gene_log_loss + (1-FLAGS.gene_log_factor) * gene_LS_loss`

`gene_non_mse_loss = FLAGS.gene_dc_factor * gene_dc_loss + (1-FLAGS.gene_dc_factor) * gene_fool_loss`

`gene_loss = FLAGS.gene_mse_factor * gene_mse_loss + (1- FLAGS.gene_mse_factor) * gene_non_mse_loss`

## Dataset
Dataset is parsed to pngs saved in specific folders
* Phantom dataset
* Knee dataset
* DCE dataset

## Results
Multiple results are exported while traning
* loss changes
* test results (zero-fill VS Recon VS Ref) saved to png for each epoch
* some of the train results are exported to png, WARNING, there was a memory bug before when we try to export all train results
* layers (some layers are skipped) of generator and detectors are exported into json after each epoch.


## Training example 
(currently working on t2)
`python srez_main.py --dataset_input /home/enhaog/GANCS/srez/dataset_MRI/phantom --batch_size 8 --run train --summary_period 123 --sample_size 256 --train_time 10  --train_dir train_save_all --R_factor 4 --R_alpha 3 --R_seed 0`              

(currently working on t2 for DCE)
`python srez_main.py --run train --dataset_input /home/enhaog/GANCS/srez/dataset_MRI/abdominal_DCE --sample_size 200 --sample_size_y 100 --sampling_pattern /home/enhaog/GANCS/srez/dataset_MRI/sampling_pattern_DCE/mask_2dvardesnity_radiaview_4fold.mat --batch_size 4  --summary_period 125 --sample_test 32 --sample_train 10000 --train_time 200  --train_dir train_DCE_test `

