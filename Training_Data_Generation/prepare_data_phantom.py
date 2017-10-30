# init
#
from scipy import io as sio
import matplotlib
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from scipy import io as sio
import h5py
from PIL import Image
# load data
# data = sio.loadmat('/home/enhaog/GANCS/srez/dataset_MRI/image_phantom_array.mat')
filename='/home/enhaog/GANCS/srez/dataset_MRI/image_phantom_array.mat'
file = h5py.File(filename,'r')
data = file['image_phantom_array']
print('data size:', data.shape)
num_data = data.shape[0]

# save png
dir_image_input = '/home/enhaog/GANCS/srez/dataset_MRI/phantom2'
try:
    os.mkdir(dir_image_input)
except:
    print('pass')

for i in xrange(num_data):
    im_input = data[i,:,:]
    # scale
    im_input = im_input-np.min(im_input[:])
    im_input = im_input/np.max(im_input[:])
    # save to image
    obj = Image.fromarray((im_input*255).astype('uint8'))
    filename_image = 'im_input_{0:05d}.jpg'.format(i)
    obj.save(os.path.join(dir_image_input, filename_image))
    if i%100 == 0:
        print('save to {}'.format(filename_image))    