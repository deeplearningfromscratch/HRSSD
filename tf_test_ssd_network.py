import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization


# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
def process_image(img):
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)

    # Input placeholder.
    net_shape = (300, 300)
    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    ssd_net = ssd_vgg_300.SSDNet()
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
    ssd_anchors = ssd_net.anchors(net_shape)

    localisations = ssd_net.bboxes_decode(localisations, ssd_anchors)

    select_threshold = 0.5
    nms_threshold = .45
    select_top_k = 400
    keep_top_k = 400
    rscores, rbboxes = ssd_net.detected_bboxes(predictions, localisations,
                                               select_threshold=select_threshold,
                                               nms_threshold=nms_threshold,
                                               clipping_bbox=None,
                                               top_k=select_top_k,
                                               keep_top_k=keep_top_k)
    # Restore SSD model.
    # ckpt_filename = 'checkpoints/ssd_300_vgg.ckpt'
    ckpt_filename = 'checkpoints/ssd_300_vgg.ckpt'
    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)

    rscores, rbboxes = isess.run([image_4d], feed_dict={img_input: img})

    return rscores, rbboxes


# SSD default anchor boxes.
# Test on some demo image and visualize output.
path = 'bdd_img/'
image_names = sorted(os.listdir(path))

img = mpimg.imread(path + image_names[-3])
rscores, rbboxes = process_image(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes_tf(img, rscores, rbboxes)
