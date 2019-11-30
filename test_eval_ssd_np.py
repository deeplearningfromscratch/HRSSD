import tensorflow as tf
slim = tf.contrib.slim
from datasets import dataset_factory
from preprocessing import ssd_vgg_preprocessing
from nets import ssd_vgg_512, np_methods

batch_size = 1
dataset_name = 'pascalvoc_2007'
dataset_split_name = 'test'
dataset_dir = 'assets/PASCAL_VOC/'
DATA_FORMAT = 'NHWC'

dataset = dataset_factory.get_dataset(
            dataset_name, dataset_split_name, dataset_dir)
ssd_shape = (512, 512)
ssd_net = ssd_vgg_512.SSDNet()
ssd_anchors = ssd_net.anchors(ssd_shape)

with tf.name_scope(dataset_name + '_data_provider'):
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity=2 * batch_size,
        common_queue_min=batch_size,
        shuffle=False)
# Get for SSD network: image, labels, bboxes.
[image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                 'object/label',
                                                 'object/bbox'])

gdifficults = tf.zeros(tf.shape(glabels), dtype=tf.int64)

# Pre-processing image, labels and bboxes.
image, glabels, gbboxes, gbbox_img = \
    ssd_vgg_preprocessing.preprocess_for_eval(image, glabels, gbboxes,
                           out_shape=ssd_shape,
                           data_format=DATA_FORMAT,
                           resize=4,
                           difficults=None)

gclasses, glocalisations, gscores = \
    ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)


with tf.Session() as sess:
    out = sess.run([image])

    print(out)



