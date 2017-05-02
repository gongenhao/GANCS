"""
train on GAN-CS 
example
export CUDA_VISIBLE_DEVICES=0
python srez_main.py --dataset_input /home/enhaog/GANCS/srez/dataset_MRI/phantom \
                    --dataset_output  /home/enhaog/GANCS/srez/dataset_MRI/phantom \
                    --run train \
                    --gene_mse_factor 1.0

python3 srez_main.py --dataset_input /home/enhaog/GANCS/srez/dataset_MRI/phantom --batch_size 8 --run train --summary_period 123 --sample_size 256 --train_time 10  --train_dir train_save_all                  
python srez_main.py --dataset_input /home/enhaog/GANCS/srez/dataset_MRI/phantom --batch_size 8 --run train --summary_period 123 --sample_size 256 --train_time 10  --train_dir train_temp_dir                  


"""
import srez_demo
import srez_input
import srez_model
import srez_train

import os.path
import random
import numpy as np
import numpy.random

import tensorflow as tf
import shutil, os, errno # utils handling file manipulation

FLAGS = tf.app.flags.FLAGS

# Configuration (alphabetically)
tf.app.flags.DEFINE_integer('batch_size', 16,
                            "Number of samples per batch.")

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 10000,
                            "Number of batches in between checkpoints")

tf.app.flags.DEFINE_string('dataset', '',
                           "Path to the dataset directory.")

tf.app.flags.DEFINE_string('dataset_output', '',
                           "Path to the dataset directory.")

tf.app.flags.DEFINE_string('dataset_input', 'dataset_input',
                           "Path to the dataset directory.")


tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_string('run', 'demo',
                            "Which operation to run. [demo|train]")

tf.app.flags.DEFINE_float('gene_log_factor', 0,
                          "Multiplier for generator fool loss term, weighting log-loss vs LS loss")

tf.app.flags.DEFINE_float('gene_dc_factor', 0.01,
                          "Multiplier for generator data-consistency L2 loss term for data consistency, weighting Data-Consistency with GD-loss for GAN-loss")

tf.app.flags.DEFINE_float('gene_mse_factor', .001,
                          "Multiplier for generator MSE loss for regression accuracy, weighting MSE VS GAN-loss")

tf.app.flags.DEFINE_float('learning_beta1', 0.5,
                          "Beta1 parameter used for AdamOptimizer")

tf.app.flags.DEFINE_float('learning_rate_start', 0.00020,
                          "Starting learning rate used for AdamOptimizer")

tf.app.flags.DEFINE_integer('learning_rate_half_life', 1000,
                            "Number of batches until learning rate is halved")

tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")

tf.app.flags.DEFINE_integer('sample_size', 128,
                            "Image sample size in pixels. Range [64,128]")

tf.app.flags.DEFINE_integer('summary_period', 500,
                            "Number of batches between summary data dumps")

tf.app.flags.DEFINE_integer('summary_train_period', 50,
                            "Number of batches between train data dumps")

tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")

tf.app.flags.DEFINE_integer('sample_test', 16,
                            "Number of features to use for testing.")
                            
tf.app.flags.DEFINE_string('train_dir', 'train',
                           "Output folder where training logs are dumped.")

tf.app.flags.DEFINE_integer('train_time', 20,
                            "Time in minutes to train the model")

tf.app.flags.DEFINE_integer('axis_undersample', 1,
                            "which axis to undersample")


def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def prepare_dirs(delete_train_dir=False, shuffle_filename=True):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    if delete_train_dir:
        try:
            if tf.gfile.Exists(FLAGS.train_dir):
                tf.gfile.DeleteRecursively(FLAGS.train_dir)
            tf.gfile.MakeDirs(FLAGS.train_dir)
        except:
            try:
                shutil.rmtree(FLAGS.train_dir)
            except:
                print('fail to delete train dir {0} using tf.gfile, will use shutil'.format(FLAGS.train_dir))
            mkdirp(FLAGS.train_dir)


    # Return names of training files
    if not tf.gfile.Exists(FLAGS.dataset) or \
       not tf.gfile.IsDirectory(FLAGS.dataset):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.dataset,))

    filenames = tf.gfile.ListDirectory(FLAGS.dataset)
    filenames = sorted(filenames)
    if shuffle_filename:
        random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.dataset, f) for f in filenames]

    return filenames

def get_filenames(dir_file='', shuffle_filename=False):
    try:
        filenames = tf.gfile.ListDirectory(dir_file)
    except:
        print('cannot get files from {0}'.format(dir_file))
        return []
    filenames = sorted(filenames)
    if shuffle_filename:
        random.shuffle(filenames)
    filenames = [os.path.join(dir_file, f) for f in filenames if f.endswith('.jpg')]
    return filenames



def setup_tensorflow(gpu_memory_fraction=0.6):
    # Create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

# SummaryWriter is deprecated
# tf.summary.FileWriter.
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    return sess, summary_writer   

def _demo():
    # Load checkpoint
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.checkpoint_dir,))

    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    filenames = prepare_dirs(delete_train_dir=False)

    # Setup async input queues
    features, labels = srez_input.setup_inputs(sess, filenames)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, features, labels)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)
    saver.restore(sess, filename)

    # Execute demo
    srez_demo.demo1(sess)

class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def _train():
    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    # all_filenames = 
    prepare_dirs(delete_train_dir=True, shuffle_filename=False)
    filenames_input = get_filenames(dir_file=FLAGS.dataset_input, shuffle_filename=False)
    # if not specify use the same as input
    if FLAGS.dataset_output == '':
        FLAGS.dataset_output = FLAGS.dataset_input
    filenames_output = get_filenames(dir_file=FLAGS.dataset_output, shuffle_filename=False)

    # Separate training and test sets
    # train_filenames = all_filenames[:-FLAGS.sample_test]
    # test_filenames  = all_filenames[-FLAGS.sample_test:]
    train_filenames_input = filenames_input[:-FLAGS.sample_test]
    test_filenames_input  = filenames_input[-FLAGS.sample_test:]
    train_filenames_output = filenames_output[:-FLAGS.sample_test]
    test_filenames_output  = filenames_output[-FLAGS.sample_test:]

    # TBD: Maybe download dataset here

    # Setup async input queues
    train_features, train_labels = srez_input.setup_inputs_one_sources(sess, train_filenames_input, train_filenames_output, 
                                                                        image_size=FLAGS.sample_size, axis_undersample=FLAGS.axis_undersample)
    test_features,  test_labels  = srez_input.setup_inputs_one_sources(sess, test_filenames_input, test_filenames_output,
                                                                        image_size=FLAGS.sample_size, axis_undersample=FLAGS.axis_undersample)
    
    # sample size
    num_sample_train = len(train_filenames_input)
    num_sample_test = len(test_filenames_input)
    print('train on {0} samples and test on {1} samples'.format(num_sample_train, num_sample_test))

    # Add some noise during training (think denoising autoencoders)
    noise_level = .00
    noisy_train_features = train_features + \
                           tf.random_normal(train_features.get_shape(), stddev=noise_level)

    # Create and initialize model
    [gene_minput, gene_moutput, \
     gene_output, gene_var_list, gene_layers, gene_mlayers, \
     disc_real_output, disc_fake_output, disc_var_list, disc_layers] = \
            srez_model.create_model(sess, noisy_train_features, train_labels)

    gene_loss, gene_dc_loss, gene_ls_loss = srez_model.create_generator_loss(disc_fake_output, gene_output, train_features, train_labels)
    disc_real_loss, disc_fake_loss = \
                     srez_model.create_discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
    
    (global_step, learning_rate, gene_minimize, disc_minimize) = \
            srez_model.create_optimizers(gene_loss, gene_var_list,
                                         disc_loss, disc_var_list)

    # Train model
    train_data = TrainData(locals())
    srez_train.train_model(train_data, num_sample_train, num_sample_test)

def main(argv=None):
    # Training or showing off?

    if FLAGS.run == 'demo':
        _demo()
    elif FLAGS.run == 'train':
        _train()

if __name__ == '__main__':
  tf.app.run()

