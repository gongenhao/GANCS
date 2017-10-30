# prepare_data_DCE.py
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

# dir
# dir_data_DCE = '/Users/enhao/Documents/Research/MRI/GANCS/data_MRI/processed_data'
# dir_mask_DCE = '/Users/enhao/Documents/Research/MRI/GANCS/data_MRI/sampling_pattern/'
# dir_image_DCE='/home/enhaog/GANCS/srez/dataset_MRI/abdominal_DCE'
dir_data_DCE = '/mnt/raid2/morteza/abdominal_DCE_processed/processed_data'
dir_mask_DCE = '/mnt/raid2/morteza/abdominal_DCE_processed/sampling_pattern'
dir_image_DCE='/home/enhaog/GANCS/srez/dataset_MRI/abdominal_DCE'

# import data
list_filename_data = [x for x in os.listdir(dir_data_DCE) if x.endswith('.mat')]
print(list_filename_data)

# generate slice images
try:
    os.mkdir(dir_image_DCE)
    print('directory {0} created'.format(dir_image_DCE))
except:
    print('directory {0} exists'.format(dir_image_DCE))

# generate images
indexes_slice=xrange(0,151)
for filename_data in list_filename_data:
    # load data
    filepath_data = os.path.join(dir_data_DCE, filename_data)
    content_mat = sio.loadmat(filepath_data)
    key_mat=[x for x in content_mat.keys() if not x.startswith('_')]
    try:
        data=content_mat[key_mat[0]]
        assert(np.ndim(data)==3)
    except:
        continue
    print('image load from {0}, size {1}'.format(filename_data, data.shape))
    # scale
    data=data/(np.max(data[:])+1e-6)    
    # each slice
    num_slice=data.shape[0]
    indexes_slice=xrange(num_slice)
    for index_slice in indexes_slice:
        data_slice = np.squeeze(data[index_slice,:,:])
        # save to image
        obj = Image.fromarray((data_slice*255).astype('uint8'))
        filename_image = '{0}_slice{1:03d}.jpg'.format(filename_data.split('.mat')[0],index_slice)
        obj.save(os.path.join(dir_image_DCE, filename_image))
        if index_slice%100 == 0:
            print('save to {}'.format(filename_image))              

print('DCE data generated to images to folder:{0}'.format(
		dir_image_DCE))