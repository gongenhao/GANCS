import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

# generate mask based on alpha
def generate_mask_alpha(size=[128,128], r_factor_designed=5.0, r_alpha=3, axis_undersample=1,
                        acs=3, seed=0, mute=0):
    # init
    mask = np.zeros(size)
    np.random.seed(seed)
    # get samples
    num_phase_encode = size[axis_undersample]
    num_phase_sampled = int(np.floor(num_phase_encode/r_factor_designed))
    # coordinate
    coordinate_normalized = np.array(xrange(num_phase_encode))
    coordinate_normalized = np.abs(coordinate_normalized-num_phase_encode/2)/(num_phase_encode/2.0)
    prob_sample = coordinate_normalized**r_alpha
    prob_sample = prob_sample/sum(prob_sample)
    # sample
    index_sample = np.random.choice(num_phase_encode, size=num_phase_sampled, 
                                    replace=False, p=prob_sample)
    # sample                
    if axis_undersample == 0:
        mask[index_sample,:]=1
    else:
        mask[:,index_sample]=1

    # acs                
    if axis_undersample == 0:
        mask[:(acs+1)/2,:]=1
        mask[-acs/2:,:]=1
    else:
        mask[:,:(acs+1)/2]=1
        mask[:,-acs/2:]=1

    # compute reduction
    r_factor = len(mask.flatten())/sum(mask.flatten())
    if not mute:
        print('gen mask for R-factor={0:.4f}'.format(r_factor))
        print(num_phase_encode, num_phase_sampled, np.where(mask[0,:]))

    return mask, r_factor


# generate mask based on .mat mask
def generate_mask_mat(mask=[], mute=0):
    # shift
    mask = np.fft.ifftshift(mask)
    # compute reduction
    r_factor = len(mask.flatten())/sum(mask.flatten())
    if not mute:
        print('load mask size of {1} for R-factor={0:.4f}'.format(r_factor, mask.shape))
    return mask, r_factor


def setup_inputs_one_sources(sess, filenames_input, filenames_output, image_size=None, 
                             axis_undersample=1, capacity_factor=3, r_factor=4, r_alpha=0, sampling_mask=None):

    # generate default mask
    if sampling_mask is None:
        DEFAULT_MASK, _ = generate_mask_alpha(image_size, # kspace size
                                              r_factor_designed=r_factor, 
                                              r_alpha=r_alpha, 
                                              axis_undersample=axis_undersample)
    else:
        # get input mask
        DEFAULT_MASK, _ = generate_mask_mat(sampling_mask)

    # convert to complex tf tensor
    DEFAULT_MAKS_TF = tf.cast(tf.constant(DEFAULT_MASK), tf.float32)
    DEFAULT_MAKS_TF_c = tf.cast(DEFAULT_MAKS_TF, tf.complex64)

    # image size
    if image_size is None:
        # image_size
        if FLAGS.sample_size_y>0:
            image_size = [FLAGS.sample_size, FLAGS.sample_size_y]
        else:
            image_size = [FLAGS.sample_size, FLAGS.sample_size]

    
    # Read each JPEG file
    reader_input = tf.WholeFileReader()
    filename_queue_input = tf.train.string_input_producer(filenames_input)
    key, value_input = reader_input.read(filename_queue_input)
    channels = 3
    image_input = tf.image.decode_jpeg(value_input, channels=channels, name="input_image")
    image_input.set_shape([None, None, channels])

    # cast image to float in 0~1
    image_input = tf.cast(image_input, tf.float32)/255.0

    # use the last channel (B) for input and output, assume image is in gray-scale
    image_output = image_input[:,:,-1]
    image_input = image_input[:,:,-1]

    # apply undersampling mask
    kspace_input = tf.fft2d(tf.cast(image_input,tf.complex64))
    kspace_zpad = kspace_input * DEFAULT_MAKS_TF_c
    # zpad undersampled image for input
    image_zpad = tf.ifft2d(kspace_zpad)
    image_zpad_real = tf.real(image_zpad)
    image_zpad_real = tf.reshape(image_zpad_real, [image_size[0], image_size[1], 1])
    image_zpad_imag = tf.imag(image_zpad)
    image_zpad_imag = tf.reshape(image_zpad_imag, [image_size[0], image_size[1], 1])    
    # concat to input, 2 channel for real and imag value
    image_zpad_concat = tf.concat(axis=2, values=[image_zpad_real, image_zpad_imag])

    # The feature is zpad image with 2 channel, label is the ground-truth real-valued image
    feature = tf.reshape(image_zpad_concat, [image_size[0], image_size[1], 2])
    label   = tf.reshape(image_output, [image_size[0], image_size[1], 1])

    # Using asynchronous queues
    features, labels = tf.train.batch([feature, label],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity = capacity_factor*FLAGS.batch_size,
                                      name='labels_and_features')

    tf.train.start_queue_runners(sess=sess)
      
    return features, labels    

