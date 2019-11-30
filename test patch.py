from nets import np_methods
import os
import matplotlib.image as mpimg

path = 'demo/'
image_names = sorted(os.listdir(path))

img = mpimg.imread(path + image_names[-4])
