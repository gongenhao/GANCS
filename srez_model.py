import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class Model:
    """A neural network model.

    Currently only supports a feedforward architecture."""
    
    def __init__(self, name, features):
        self.name = name
        self.outputs = [features]

    def _get_layer_str(self, layer=None):
        if layer is None:
            layer = self.get_num_layers()
        
        return '%s_L%03d' % (self.name, layer+1)

    def _get_num_inputs(self):
        return int(self.get_output().get_shape()[-1])

    def _glorot_initializer(self, prev_units, num_units, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
        stddev  = np.sqrt(stddev_factor / np.sqrt(prev_units*num_units))
        return tf.truncated_normal([prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def _glorot_initializer_conv2d(self, prev_units, num_units, mapsize, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""

        stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def get_num_layers(self):
        return len(self.outputs)

    def add_batch_norm(self, scale=False):
        """Adds a batch normalization layer to this model.

        See ArXiv 1502.03167v3 for details."""

        # TBD: This appears to be very flaky, often raising InvalidArgumentError internally
        with tf.variable_scope(self._get_layer_str()):
            out = tf.contrib.layers.batch_norm(self.get_output(), scale=scale)
        
        self.outputs.append(out)
        return self

    def add_flatten(self):
        """Transforms the output of this network to a 1D tensor"""

        with tf.variable_scope(self._get_layer_str()):
            batch_size = int(self.get_output().get_shape()[0])
            out = tf.reshape(self.get_output(), [batch_size, -1])

        self.outputs.append(out)
        return self

    def add_dense(self, num_units, stddev_factor=1.0):
        """Adds a dense linear layer to this model.

        Uses Glorot 2010 initialization assuming linear activation."""
        
        assert len(self.get_output().get_shape()) == 2, "Previous layer must be 2-dimensional (batch, channels)"

        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            
            # Weight term
            initw   = self._glorot_initializer(prev_units, num_units,
                                               stddev_factor=stddev_factor)
            weight  = tf.get_variable('weight', initializer=initw)

            # Bias term
            initb   = tf.constant(0.0, shape=[num_units])
            bias    = tf.get_variable('bias', initializer=initb)

            # Output of this layer
            out     = tf.matmul(self.get_output(), weight) + bias

        self.outputs.append(out)
        return self

    def add_sigmoid(self):
        """Adds a sigmoid (0,1) activation function layer to this model."""

        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            out = tf.nn.sigmoid(self.get_output())
        
        self.outputs.append(out)
        return self

    def add_softmax(self):
        """Adds a softmax operation to this model"""

        with tf.variable_scope(self._get_layer_str()):
            this_input = tf.square(self.get_output())
            reduction_indices = list(range(1, len(this_input.get_shape())))
            acc = tf.reduce_sum(this_input, reduction_indices=reduction_indices, keep_dims=True)
            out = this_input / (acc+FLAGS.epsilon)
            #out = tf.verify_tensor_all_finite(out, "add_softmax failed; is sum equal to zero?")
        
        self.outputs.append(out)
        return self

    def add_relu(self):
        """Adds a ReLU activation function to this model"""

        with tf.variable_scope(self._get_layer_str()):
            out = tf.nn.relu(self.get_output())

        self.outputs.append(out)
        return self        

    def add_elu(self):
        """Adds a ELU activation function to this model"""

        with tf.variable_scope(self._get_layer_str()):
            out = tf.nn.elu(self.get_output())

        self.outputs.append(out)
        return self      

    def add_lrelu(self, leak=.2):
        """Adds a leaky ReLU (LReLU) activation function to this model"""

        with tf.variable_scope(self._get_layer_str()):
            t1  = .5 * (1 + leak)
            t2  = .5 * (1 - leak)
            out = t1 * self.get_output() + \
                  t2 * tf.abs(self.get_output())
            
        self.outputs.append(out)
        return self

    def add_conv2d(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a 2D convolutional layer."""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"
        
        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            out    = tf.nn.conv2d(self.get_output(), weight,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')

            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self

    def add_conv2d_transpose(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a transposed 2D convolutional layer"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        with tf.variable_scope(self._get_layer_str()):
            prev_units = self._get_num_inputs()
            
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('weight', initializer=initw)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2])
            prev_output = self.get_output()
            output_shape = [FLAGS.batch_size,
                            int(prev_output.get_shape()[1]) * stride,
                            int(prev_output.get_shape()[2]) * stride,
                            num_units]
            out    = tf.nn.conv2d_transpose(self.get_output(), weight,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding='SAME')

            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self

    def add_residual_block(self, num_units, mapsize=3, num_layers=2, stddev_factor=1e-3):
        """Adds a residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]):
            self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=1.)

        bypass = self.get_output()

        # Residual block
        for _ in range(num_layers):
            self.add_batch_norm()
            self.add_relu()
            self.add_conv2d(num_units, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)

        self.add_sum(bypass)

        return self

    def add_bottleneck_residual_block(self, num_units, mapsize=3, stride=1, transpose=False):
        """Adds a bottleneck residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]) or stride != 1:
            ms = 1 if stride == 1 else mapsize
            #bypass.add_batch_norm() # TBD: Needed?
            if transpose:
                self.add_conv2d_transpose(num_units, mapsize=ms, stride=stride, stddev_factor=1.)
            else:
                self.add_conv2d(num_units, mapsize=ms, stride=stride, stddev_factor=1.)

        bypass = self.get_output()

        # Bottleneck residual block
        self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units//4, mapsize=1,       stride=1,      stddev_factor=2.)

        self.add_batch_norm()
        self.add_relu()
        if transpose:
            self.add_conv2d_transpose(num_units//4,
                                      mapsize=mapsize,
                                      stride=1,
                                      stddev_factor=2.)
        else:
            self.add_conv2d(num_units//4,
                            mapsize=mapsize,
                            stride=1,
                            stddev_factor=2.)

        self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units,    mapsize=1,       stride=1,      stddev_factor=2.)

        self.add_sum(bypass)

        return self

    def add_sum(self, term):
        """Adds a layer that sums the top layer with the given term"""

        with tf.variable_scope(self._get_layer_str()):
            prev_shape = self.get_output().get_shape()
            term_shape = term.get_shape()
            #print("%s %s" % (prev_shape, term_shape))
            assert prev_shape == term_shape and "Can't sum terms with a different size"
            out = tf.add(self.get_output(), term)
        
        self.outputs.append(out)
        return self

    def add_mean(self):
        """Adds a layer that averages the inputs from the previous layer"""

        with tf.variable_scope(self._get_layer_str()):
            prev_shape = self.get_output().get_shape()
            reduction_indices = list(range(len(prev_shape)))
            assert len(reduction_indices) > 2 and "Can't average a (batch, activation) tensor"
            reduction_indices = reduction_indices[1:-1]
            out = tf.reduce_mean(self.get_output(), reduction_indices=reduction_indices)
        
        self.outputs.append(out)
        return self

    def add_upscale(self):
        """Adds a layer that upscales the output by 2x through nearest neighbor interpolation"""

        prev_shape = self.get_output().get_shape()
        size = [2 * int(s) for s in prev_shape[1:3]]
        out  = tf.image.resize_nearest_neighbor(self.get_output(), size)

        self.outputs.append(out)
        return self        

    def get_output(self):
        """Returns the output from the topmost layer of the network"""
        return self.outputs[-1]

    def get_variable(self, layer, name):
        """Returns a variable given its layer and name.

        The variable must already exist."""

        scope      = self._get_layer_str(layer)
        collection = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

        # TBD: Ugly!
        for var in collection:
            if var.name[:-2] == scope+'/'+name:
                return var

        return None

    def get_all_layer_variables(self, layer):
        """Returns all variables in the given layer"""
        scope = self._get_layer_str(layer)
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

def _discriminator_model(sess, features, disc_input, layer_output_skip=3):
    # Fully convolutional model
    mapsize = 3
    layers  = [8, 16, 32, 64]#[64, 128, 256, 512]

    old_vars = tf.global_variables()#tf.all_variables() , all_variables() are deprecated

    model = Model('DIS', 2*disc_input - 1)

    for layer in range(len(layers)):
        nunits = layers[layer]
        stddev_factor = 2.0

        model.add_conv2d(nunits, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
        model.add_batch_norm()
        model.add_relu()

    # Finalization a la "all convolutional net"
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_batch_norm()
    model.add_relu()

    # Linearly map to real/fake and return average score
    # (softmax will be applied later)
    model.add_conv2d(1, mapsize=1, stride=1, stddev_factor=stddev_factor)
    model.add_mean()

    new_vars  = tf.global_variables()#tf.all_variables() , all_variables() are deprecated
    disc_vars = list(set(new_vars) - set(old_vars))

    #select output
    output_layers = [model.outputs[0]] + model.outputs[1:layer_output_skip:-1] + [model.outputs[-1]]

    return model.get_output(), disc_vars, output_layers

# def _generator_model(sess, features, labels, channels):
#     # Upside-down all-convolutional resnet

#     mapsize = 3
#     res_units  = [256, 128, 96]

#     old_vars = tf.global_variables()#tf.all_variables() , all_variables() are deprecated

#     # See Arxiv 1603.05027
#     model = Model('GEN', features)

#     for ru in range(len(res_units)-1):
#         nunits  = res_units[ru]

#         for j in range(2):
#             model.add_residual_block(nunits, mapsize=mapsize)

#         # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
#         # and transposed convolution
#         # model.add_upscale()
        
#         model.add_batch_norm()
#         model.add_relu()
#         model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

#     # Finalization a la "all convolutional net"
#     nunits = res_units[-1]
#     model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
#     # Worse: model.add_batch_norm()
#     model.add_relu()

#     model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=2.)
#     # Worse: model.add_batch_norm()
#     model.add_relu()

#     # Last layer is sigmoid with no batch normalization
#     model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)
#     model.add_sigmoid()
    
#     new_vars  = tf.global_variables()#tf.all_variables() , all_variables() are deprecated
#     gene_vars = list(set(new_vars) - set(old_vars))

#     return model.get_output(), gene_vars

def conv(batch_input, out_channels, stride=2, size_kernel=4):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [size_kernel, size_kernel, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

def deconv(batch_input, out_channels, size_kernel=4):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [size_kernel, size_kernel, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv        

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized      


def _generator_encoder_decoder(sess, features, labels, channels, layer_output_skip=3):

    # old variables
    layers = []    
    old_vars = tf.global_variables()#tf.all_variables() , all_variables() are deprecated
    # layers.append(features)

    # definition
    num_filter_generator = 8
    layer_specs = [ 
        num_filter_generator * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        num_filter_generator * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        num_filter_generator * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        # num_filter_generator * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        # num_filter_generator * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        # num_filter_generator * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        num_filter_generator * 16, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

   # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(features, num_filter_generator, stride=2)
        layers.append(output)

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        # (num_filter_generator * 16, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        # (num_filter_generator * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        # (num_filter_generator * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (num_filter_generator * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (num_filter_generator * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (num_filter_generator * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (num_filter_generator, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    for x in layers:
        print(x)

    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, channels)
        output = tf.tanh(output)
        layers.append(output)


    # out variables
    new_vars  = tf.global_variables()#tf.all_variables() , all_variables() are deprecated
    gene_vars = list(set(new_vars) - set(old_vars))

    # select subset of layers
    output_layers = [layers[0]] + layers[1:layer_output_skip:-1] + [layers[-1]]

    return layers[-1], gene_vars, output_layers


def _generator_model_with_scale(sess, features, labels, channels, layer_output_skip=3):
    # Upside-down all-convolutional resnet

    mapsize = 3
    res_units  = [64, 32, 16]#[256, 128, 96]
    scale_changes = [0,0,0,0,0,0]

    old_vars = tf.global_variables()#tf.all_variables() , all_variables() are deprecated

    # See Arxiv 1603.05027
    model = Model('GEN', features)

    for ru in range(len(res_units)-1):
        nunits  = res_units[ru]

        for j in range(2):
            model.add_residual_block(nunits, mapsize=mapsize)

        # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
        # and transposed convolution
        if scale_changes[ru]>0:
            model.add_upscale()

        model.add_batch_norm()
        model.add_relu()
        model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)


    # Finalization a la "all convolutional net"
    nunits = res_units[-1]
    model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
    # Worse: model.add_batch_norm()
    model.add_relu()

    model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=2.)
    # Worse: model.add_batch_norm()
    model.add_relu()

    # Last layer is sigmoid with no batch normalization
    model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)
    model.add_sigmoid()
    
    new_vars  = tf.global_variables()#tf.all_variables() , all_variables() are deprecated
    gene_vars = list(set(new_vars) - set(old_vars))

    # select subset of layers
    output_layers = [model.outputs[0]] + model.outputs[1:layer_output_skip:-1] + [model.outputs[-1]]

    return model.get_output(), gene_vars, output_layers

def create_model(sess, features, labels, architecture='resnet'):
    # sess: TF sesson
    # features: input, for SR/CS it is the input image
    # labels: output, for SR/CS it is the groundtruth image
    # architecture: aec for encode-decoder, resnet for upside down 
    # Generator
    rows      = int(features.get_shape()[1])
    cols      = int(features.get_shape()[2])
    channels  = int(features.get_shape()[3])

    gene_minput = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, rows, cols, channels])

    # TBD: Is there a better way to instance the generator?
    if architecture == 'aec':
        function_generator = lambda x,y,z,w: _generator_encoder_decoder(x,y,z,w)
    else:
        function_generator = lambda x,y,z,w: _generator_model_with_scale(x,y,z,w)
    with tf.variable_scope('gene') as scope:
        gene_output, gene_var_list, gene_layers = function_generator(sess, features, labels, 1)
                    # _generator_model_with_scale(sess, features, labels, 1)

        scope.reuse_variables()

        # for testing input
        gene_moutput, _ , gene_mlayers= function_generator(sess, gene_minput, labels, 1)
                    # _generator_model_with_scale(sess, gene_minput, labels, 1)
    
    # Discriminator with real data
    disc_real_input = tf.identity(labels, name='disc_real_input')

    # TBD: Is there a better way to instance the discriminator?
    with tf.variable_scope('disc') as scope:
        disc_real_output, disc_var_list, disc_layers = \
                _discriminator_model(sess, features, disc_real_input)

        scope.reuse_variables()
            
        disc_fake_output, _, _ = _discriminator_model(sess, features, gene_output)

    return [gene_minput,      gene_moutput,
            gene_output,      gene_var_list, gene_layers, gene_mlayers,
            disc_real_output, disc_fake_output, disc_var_list, disc_layers]
            

def _downscale(images, K):
    """Differentiable image downscaling by a factor of K"""
    arr = np.zeros([K, K, 3, 3])
    arr[:,:,0,0] = 1.0/(K*K)
    arr[:,:,1,1] = 1.0/(K*K)
    arr[:,:,2,2] = 1.0/(K*K)
    dowscale_weight = tf.constant(arr, dtype=tf.float32)
    
    downscaled = tf.nn.conv2d(images, dowscale_weight,
                              strides=[1, K, K, 1],
                              padding='SAME')
    return downscaled

# def create_generator_loss(disc_output, gene_output, features):
#     # I.e. did we fool the discriminator?
#     cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(disc_output, tf.ones_like(disc_output))
#     gene_ce_loss  = tf.reduce_mean(cross_entropy, name='gene_ce_loss')

#     # I.e. does the result look like the feature?
#     K = int(gene_output.get_shape()[1])//int(features.get_shape()[1])
#     assert K == 2 or K == 4 or K == 8    
#     downscaled = _downscale(gene_output, K)
    
#     gene_l1_loss  = tf.reduce_mean(tf.abs(downscaled - features), name='gene_l1_loss')

#     gene_loss     = tf.add((1.0 - FLAGS.gene_l1_factor) * gene_ce_loss,
#                            FLAGS.gene_l1_factor * gene_l1_loss, name='gene_loss')
    
#     return gene_loss

def Fourier(x, separate_complex=True):    
    x = tf.cast(x, tf.complex64)
    if separate_complex:
        x_complex = x[:,:,:,0]+1j*x[:,:,:,1]
    else:
        x_complex = x
    x_complex = tf.reshape(x_complex,x_complex.get_shape()[:3])
    y_complex = tf.fft2d(x_complex)
    print('using Fourier, input dim {0}, output dim {1}'.format(x.get_shape(), y_complex.get_shape()))
    # x = tf.cast(x, tf.complex64)
    # y = tf.fft3d(x)
    # y = y[:,:,:,-1]
    return y_complex


def create_generator_loss(disc_output, gene_output, features, labels):
    # I.e. did we fool the discriminator?
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_output, labels=tf.ones_like(disc_output))
    gene_ce_loss  = tf.reduce_mean(cross_entropy, name='gene_ce_loss')

    # LS cost
    ls_loss = tf.square(disc_output - tf.ones_like(disc_output))
    gene_ls_loss  = tf.reduce_mean(ls_loss, name='gene_ls_loss')

    # I.e. does the result look like the feature?
    # K = int(gene_output.get_shape()[1])//int(features.get_shape()[1])
    # assert K == 2 or K == 4 or K == 8    
    # downscaled = _downscale(gene_output, K)

    # fourier_transform
    gene_kspace = Fourier(gene_output, separate_complex=False)
    feature_kspace = Fourier(features, separate_complex=True)
    
    # mask to get affine projection error
    threshold_zero = 1./255.
    feature_mask = tf.greater(tf.abs(feature_kspace),threshold_zero)
    print('mask shape , get_shape():', feature_mask.get_shape())

    loss_kspace = tf.cast(tf.abs(tf.square(gene_kspace - feature_kspace)),tf.float32)*tf.cast(feature_mask,tf.float32)
    print('loss_kspace shape , get_shape():', loss_kspace.get_shape())


    # compare with real output
    print('real output , get_shape():', labels.get_shape())
        
    # data consistency
    gene_dc_loss  = tf.reduce_mean(loss_kspace, name='gene_dc_loss')
    # mse loss
    gene_mse_loss = tf.reduce_mean(tf.square(gene_output - labels), name='gene_mse_loss')
    
    # generator fool descriminator loss: gan LS or log loss
    gene_fool_loss = tf.add((1.0 - FLAGS.gene_log_factor) * gene_ls_loss,
                           FLAGS.gene_log_factor * gene_ce_loss, name='gene_fool_loss')

    # non-mse loss = fool-loss + data consisntency loss
    gene_non_mse_l2     = tf.add((1.0 - FLAGS.gene_dc_factor) * gene_fool_loss,
                           FLAGS.gene_dc_factor * gene_dc_loss, name='gene_nonmse_l2')
    
    #total loss = fool-loss + data consistency loss + mse forward-passing loss
    gene_loss     = tf.add((1.0 - FLAGS.gene_mse_factor) * gene_non_mse_l2, 
                            FLAGS.gene_mse_factor * gene_mse_loss, name='gene_loss')
    
    return gene_loss, gene_dc_loss, gene_ls_loss
    

def create_discriminator_loss(disc_real_output, disc_fake_output):
    # I.e. did we correctly identify the input as real or not?
    # cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_output, labels=tf.ones_like(disc_real_output))
    # disc_real_loss     = tf.reduce_mean(cross_entropy_real, name='disc_real_loss')
    
    # cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_output, labels=tf.zeros_like(disc_fake_output))
    # disc_fake_loss     = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')

    # ls loss
    ls_loss_real = tf.square(disc_real_output - tf.ones_like(disc_real_output))
    disc_real_loss = tf.reduce_mean(ls_loss_real, name='disc_real_loss')

    ls_loss_fake = tf.square(disc_fake_output)
    disc_fake_loss = tf.reduce_mean(ls_loss_fake, name='disc_fake_loss')


    return disc_real_loss, disc_fake_loss

def create_optimizers(gene_loss, gene_var_list,
                      disc_loss, disc_var_list):    
    # TBD: Does this global step variable need to be manually incremented? I think so.
    global_step    = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')
    learning_rate  = tf.placeholder(dtype=tf.float32, name='learning_rate')
    
    gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=FLAGS.learning_beta1,
                                       name='gene_optimizer')
    disc_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=FLAGS.learning_beta1,
                                       name='disc_optimizer')

    gene_minimize = gene_opti.minimize(gene_loss, var_list=gene_var_list, name='gene_loss_minimize', global_step=global_step)
    
    disc_minimize     = disc_opti.minimize(disc_loss, var_list=disc_var_list, name='disc_loss_minimize', global_step=global_step)
    
    return (global_step, learning_rate, gene_minimize, disc_minimize)



