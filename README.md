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

## Loss formulation
By default

`FLAGS.gene_log_factor = 0 # log loss vs least-square loss`

`FLAGS.gene_dc_factor = 0.9 # data-consistency (kspace) loss vs generator loss`

`FLAGS.gene_mse_factor = 0.001 # simple MSE loss originally for forward-passing model vs GAN loss`
GAN loss = generator loss + discriminator loss

`gene_fool_loss = FLAGS.gene_log_factor * gene_log_loss + (1-FLAGS.gene_log_factor) * gene_LS_loss`

`gene_non_mse_loss = FLAGS.gene_dc_factor * gene_dc_loss + (1-FLAGS.gene_dc_factor) * gene_fool_loss`

`gene_loss = FLAGS.gene_mse_factor * gene_mse_loss + (1- FLAGS.gene_mse_factor) * gene_non_mse_loss`

## References
https://github.com/david-gpu/srez
https://arxiv.org/abs/1609.04802
https://github.com/carpedm20/DCGAN-tensorflow
http://wiseodd.github.io/techblog/2017/03/02/least-squares-gan/
http://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/
