# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a SSD model
on a given dataset."""
import math
import sys
import six
import time

import numpy as np
import tensorflow as tf
import tf_extended as tfe
import tf_utils
from tensorflow.python.framework import ops

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from nets import np_methods

slim = tf.contrib.slim
tf.enable_eager_execution()
# =========================================================================== #
# Some default EVAL parameters
# =========================================================================== #
# List of recalls values at which precision is evaluated.
LIST_RECALLS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
                0.90, 0.95, 0.96, 0.97, 0.98, 0.99]
DATA_FORMAT = 'NHWC'

# =========================================================================== #
# SSD evaluation Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'select_threshold', 0.01, 'Selection threshold.')
tf.app.flags.DEFINE_integer(
    'select_top_k', 400, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer(
    'keep_top_k', 200, 'Keep top-k detected objects.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.45, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_float(
    'matching_threshold', 0.5, 'Matching threshold with groundtruth objects.')
tf.app.flags.DEFINE_integer(
    'eval_resize', 4, 'Image resizing: None / CENTRAL_CROP / PAD_AND_RESIZE / WARP_RESIZE.')
tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size.')
tf.app.flags.DEFINE_boolean(
    'remove_difficult', True, 'Remove difficult objects from evaluation.')

# =========================================================================== #
# Main evaluation flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', 'checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc_2007', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', 'assets/PASCAL_VOC/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'model_name', 'ssd_300_vgg', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.1, 'GPU memory fraction to use.')
tf.app.flags.DEFINE_boolean(
    'wait_for_checkpoints', False, 'Wait for new checkpoints in the eval loop.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        # =================================================================== #
        # Dataset + SSD model + Pre-processing
        # =================================================================== #
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        # Get the SSD network and its anchors.
        ssd_class = nets_factory.get_network(FLAGS.model_name)
        ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_class(ssd_params)

        # Evaluation shape and associated anchors: eval_image_size
        net_shape = (300, 300)
        ssd_shape = net_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)

        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=False)

        tf_utils.print_configuration(FLAGS.__flags, ssd_params,
                                     dataset.data_sources, FLAGS.eval_dir)
        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        with tf.device('/cpu:0'):
            with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    common_queue_capacity=2 * FLAGS.batch_size,
                    common_queue_min=FLAGS.batch_size,
                    shuffle=False)
            # Get for SSD network: image, labels, bboxes.
            [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                             'object/label',
                                                             'object/bbox'])
            if FLAGS.remove_difficult:
                [gdifficults] = provider.get(['object/difficult'])
            else:
                gdifficults = tf.zeros(tf.shape(glabels), dtype=tf.int64)

            # Pre-processing image, labels and bboxes.

            gpu_options = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
            isess = tf.InteractiveSession(config=config)

            net_shape = (300, 300)
            data_format = 'NHWC'
            img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

            image_pre, glabels, gbboxes, gbbox_img = \
                image_preprocessing_fn(image, glabels, gbboxes,
                                       out_shape=ssd_shape,
                                       data_format=DATA_FORMAT,
                                       resize=FLAGS.eval_resize,
                                       difficults=None)
            image_4d = tf.expand_dims(image_pre, 0)

            from nets import ssd_vgg_300

            ssd_net = ssd_vgg_300.SSDNet()
            with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
                rpredictions, rlocalisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=tf.AUTO_REUSE)
            print('yes')
            ckpt_filename = 'checkpoints/ssd_300_vgg.ckpt'
            isess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(isess, ckpt_filename)

            ssd_anchors = ssd_net.anchors(net_shape)
            bbox_img = gbbox_img

            select_threshold = 0.5
            nms_threshold = .45
            net_shape = (300, 300)
            # Run SSD network.

            rbbox_img = gbbox_img

            print(image_4d)
            # rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, rpredictions, rlocalisations, gbbox_img])
            # rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            #     rpredictions, rlocalisations, ssd_anchors,
            #     select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
            #
            #
            # rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
            # rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
            # rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
            # # Resize bboxes to original image shape. Note: useless for Resize.WARP!
            # rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)



def np_ssd_bboxes_select(rpredictions, rlocalisations, net_shape):
    return np_methods.ssd_bboxes_select(rpredictions, rlocalisations, ssd_anchors,
                                        select_threshold=FLAGS.select_threshold, img_shape=net_shape, num_classes=21,
                                        decode=True)


def py_postprocess(rpredictions, rlocalisations, ssd_anchors, net_shape, rbbox_img):
    from nets import np_methods
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=FLAGS.select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=FLAGS.nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

    return rscores, rbboxes


if __name__ == '__main__':
    tf.app.run()
