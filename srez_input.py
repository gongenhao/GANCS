import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def setup_inputs(sess, filenames, image_size=None, capacity_factor=3):

    if image_size is None:
        image_size = FLAGS.sample_size
    
    # Read each JPEG file
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    channels = 3
    image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")
    image.set_shape([None, None, channels])

    # Crop and other random augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, .95, 1.05)
    image = tf.image.random_brightness(image, .05)
    image = tf.image.random_contrast(image, .95, 1.05)

    wiggle = 8
    off_x, off_y = 25-wiggle, 60-wiggle
    crop_size = FLAGS.sample_size
    crop_size_plus = crop_size + 2*wiggle
    image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)
    image = tf.random_crop(image, [crop_size, crop_size, 3])

    image = tf.reshape(image, [1, crop_size, crop_size, 3])
    image = tf.cast(image, tf.float32)/255.0 # from 0~1

    if crop_size != image_size:
        image = tf.image.resize_area(image, [image_size, image_size])
 
    # The feature is simply a Kx downscaled version
    K = 4
    downsampled = tf.image.resize_area(image, [image_size//K, image_size//K])

    feature = tf.reshape(downsampled, [image_size//K, image_size//K, 3])
    label   = tf.reshape(image,       [image_size,   image_size,     3])

    # Using asynchronous queues
    features, labels = tf.train.batch([feature, label],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity = capacity_factor*FLAGS.batch_size,
                                      name='labels_and_features')

    tf.train.start_queue_runners(sess=sess)
      
    return features, labels

def setup_inputs_two_sources(sess, filenames_input, filenames_output, image_size=None, capacity_factor=3):

    if image_size is None:
        image_size = FLAGS.sample_size
    
    # Read each JPEG file
    reader_input = tf.WholeFileReader()
    filename_queue_input = tf.train.string_input_producer(filenames_input)
    key, value_input = reader_input.read(filename_queue_input)
    channels = 3
    image_input = tf.image.decode_jpeg(value_input, channels=channels, name="input_image")
    image_input.set_shape([None, None, channels])

    # read output
    reader_output = tf.WholeFileReader()
    filename_queue_output = tf.train.string_input_producer(filenames_output)
    key, value_output = reader_output.read(filename_queue_output)
    channels = 3
    image_output = tf.image.decode_jpeg(value_output, channels=channels, name="output_image")
    image_output.set_shape([None, None, channels])

    # cast
    image_input = tf.cast(image_input, tf.float32)/255.0
    image_output = tf.cast(image_output, tf.float32)/255.0

    # do undersampling here

    # take channel0 real part, channel1 imag part    
    image_input = image_input[:,:,:2]
    image_output = image_output[:,:,0]

    # The feature is simply a Kx downscaled version
    feature = tf.reshape(image_input, [image_size, image_size, 2])
    label   = tf.reshape(image_output, [image_size,   image_size,     1])

    # Using asynchronous queues
    features, labels = tf.train.batch([feature, label],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity = capacity_factor*FLAGS.batch_size,
                                      name='labels_and_features')

    tf.train.start_queue_runners(sess=sess)
      
    return features, labels

# generate mask based on alpha
def generate_mask_alpha(size=[128,128], r_factor_designed=5.0, r_alpha=3, acs=3, seed=0, axis_undersample=1, mute=0):
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


def setup_inputs_one_sources(sess, filenames_input, filenames_output, image_size=None, 
                             axis_undersample=1, capacity_factor=3, r_factor=4, r_alpha=0, undersample_mask=None):

    # generate default mask
    if undersample_mask is None:
        DEFAULT_MASK, _ = generate_mask_alpha([image_size,image_size], # kspace size
                                              r_factor_designed=r_factor, 
                                              r_alpha=r_alpha, 
                                              axis_undersample=axis_undersample)
    else:
        DEFAULT_MASK = undersample_mask
    DEFAULT_MAKS_TF = tf.cast(tf.constant(DEFAULT_MASK), tf.float32)
    DEFAULT_MAKS_TF_c = tf.cast(DEFAULT_MAKS_TF, tf.complex64)

    if image_size is None:
        image_size = FLAGS.sample_size
    
    # Read each JPEG file
    reader_input = tf.WholeFileReader()
    filename_queue_input = tf.train.string_input_producer(filenames_input)
    key, value_input = reader_input.read(filename_queue_input)
    channels = 3
    image_input = tf.image.decode_jpeg(value_input, channels=channels, name="input_image")
    image_input.set_shape([None, None, channels])

    # cast
    image_input = tf.cast(image_input, tf.float32)/255.0

    # take channel0 real part, channel1 imag part    
    image_output = image_input[:,:,-1]
    image_input = image_input[:,:,-1]
    

    # undersample here
    kspace_input = tf.fft2d(tf.cast(image_input,tf.complex64))
    kspace_zpad = kspace_input * DEFAULT_MAKS_TF_c
    image_zpad = tf.ifft2d(kspace_zpad)
    image_zpad_real = tf.real(image_zpad)
    image_zpad_real = tf.reshape(image_zpad_real, [image_size, image_size, 1])
    # image_zpad_real.set_shape([image_size, image_size, 1])
    image_zpad_imag = tf.imag(image_zpad)
    image_zpad_imag = tf.reshape(image_zpad_imag, [image_size, image_size, 1])    
    # image_zpad_imag.set_shape([image_size, image_size, 1])
    image_zpad_concat = tf.concat(axis=2, values=[image_zpad_real, image_zpad_imag])


    # The feature is simply a Kx downscaled version
    feature = tf.reshape(image_zpad_concat, [image_size, image_size, 2])
    label   = tf.reshape(image_output, [image_size,   image_size,     1])

    # Using asynchronous queues
    features, labels = tf.train.batch([feature, label],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity = capacity_factor*FLAGS.batch_size,
                                      name='labels_and_features')

    tf.train.start_queue_runners(sess=sess)
      
    return features, labels    

