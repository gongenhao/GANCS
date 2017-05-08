# GANCS
Compressed Sensing MRI based on Generative Adversarial Network 

A modification from srez super resolution github repo. https://github.com/david-gpu/srez

## descriminator related loss
Currently we are trying to explore multiple kinds of GAN loss include:
* log-loss
* LS loss
* Cycle-GAN loss (todo)
* WGAN (todo)
* based on MR priors, such as GAN on both image and k-space

## Generator Model
### Model from super-resolution paper

### Convolutional Encoder-Decoder with bypasses


## Descriminator Model
Currently we are using 4*(Conv-BN-RELU-POOL)+2*(CONV-BN-RELU)+CONV+MEAN+softmax (for logloss)

## Loss formulation
By default

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
`python srez_main.py --dataset_input /home/enhaog/GANCS/srez/dataset_MRI/phantom --batch_size 8 --run train --summary_period 123 --sample_size 256 --train_time 10  --train_dir train_save_all --R_factor 4 --R_alpha 3`              

(currently working on t2 for DCE)
`python srez_main.py --run train --dataset_input /home/enhaog/GANCS/srez/dataset_MRI/abdominal_DCE --sample_size 200 --sample_size_y 100 --sampling_pattern /home/enhaog/GANCS/srez/dataset_MRI/sampling_pattern_DCE/mask_2dvardesnity_radiaview_4fold.mat --batch_size 4  --summary_period 125 --sample_test 32 --sample_train 10000 --train_time 200  --train_dir train_DCE_test `

## References
https://github.com/david-gpu/srez
https://arxiv.org/abs/1609.04802
https://github.com/carpedm20/DCGAN-tensorflow
http://wiseodd.github.io/techblog/2017/03/02/least-squares-gan/
http://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/
