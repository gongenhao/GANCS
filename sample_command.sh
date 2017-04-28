python3 srez_main.py --dataset_input /home/enhaog/GANCS/srez/dataset_MRI/phantom --dataset_output  /home/enhaog/GANCS/srez/dataset_MRI/phantom  --batch_size 8 --run train --gene_mse_factor 0.1 --summary_period 125 --sample_size 256 --train_time 10                    

python3 srez_main.py --dataset_input /home/enhaog/GANCS/srez/dataset_MRI/phantom --dataset_output  /home/enhaog/GANCS/srez/dataset_MRI/phantom  --batch_size 8 --run train --gene_mse_factor 0.01 --gene_l2_factor 0.1 --summary_period 125 --sample_size 256 --train_time 100
