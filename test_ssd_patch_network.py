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
from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)


# Input placeholder.

def inference(net_shape, isess, num_use=None):
    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    reuse = tf.AUTO_REUSE
    ssd_net = ssd_vgg_512.SSDNet()
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

    predictions = predictions[:num_use]
    localisations = localisations[:num_use]
    # Restore SSD model.
    # ckpt_filename = 'checkpoints/ssd_300_vgg.ckpt'
    # ckpt_filename = 'checkpoints/VGG_VOC0712_SSD_300x300_iter_120000.ckpt'
    ckpt_filename = 'checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt'
    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)

    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_shape)

    return img_input, image_4d, predictions, localisations, bbox_img, ssd_anchors


# Main image processing routine.
def process_image_non_patch(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.

    input_shape = (1024, 1024)
    patch_shape = (540, 540)
    resized_img = np_methods.resize_test_image(img, input_shape)
    img_patches = np_methods.slice_input_image_into_patches(resized_img, patch_shape)
    crop_areas = np_methods.get_cropping_regions_for_patches_numpy([0, 0, input_shape[0], input_shape[1]], patch_shape)

    classes = []
    scores = []
    bboxes = []
    isess = tf.InteractiveSession(config=config)

    img_input, image_4d, predictions, localisations, bbox_img, ssd_anchors = inference((512, 512), isess)
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    # rbboxes = np_methods.bboxes_calibrate(rbboxes, input_shape,
    #                                       (300, 300), input_shape)
    classes.append(rclasses)
    scores.append(rscores)
    bboxes.append(rbboxes)

    for img, crop_area in zip(img_patches, crop_areas):
        img_input, image_4d, predictions, localisations, bbox_img, ssd_anchors = inference((512, 512), isess,
                                                                                           num_use=-3)
        rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                                  feed_dict={img_input: img})

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)

        rbboxes = np_methods.bboxes_calibrate(rbboxes, input_shape,
                                              patch_shape, crop_area)
        classes.append(rclasses)
        scores.append(rscores)
        bboxes.append(rbboxes)

    cut_outs = list(np_methods.cut_tensor_by_thresholding(classes, scores, bboxes, filter_threshold=0.7))

    flattened_netouts = list(map(lambda args: np_methods.flatten_net_outs(*args), cut_outs))

    # print(flattened_netouts)

    sorted_outs = list(map(lambda args: np_methods.bboxes_sort(*args, top_k=400), \
                           flattened_netouts))

    classes, scores, bboxes = map(lambda tup: np.concatenate(tup, axis=0), zip(*sorted_outs))

    bboxes = np_methods.bboxes_clip(bbox_ref=[0., 0., 1., 1.], bboxes=bboxes)
    classes, scores, bboxes = np_methods.bboxes_nms(classes,
                                                    scores,
                                                    bboxes,
                                                    nms_threshold=nms_threshold)
    # return classes, scores, bboxes
    # box_merging logic
    patch_areas = np_methods.get_normalized_patch_areas(input_shape, crop_areas)
    rclasses, rscores, rbboxes = np_methods.splitted_bboxes_nms(classes, scores, bboxes, patch_areas, nms_threshold=0.15)

    return rclasses, rscores, rbboxes


# Test on some demo image and visualize output.
path = 'demo/'
image_names = sorted(os.listdir(path))
print(image_names)

img = mpimg.imread(path + image_names[3])
rclasses, rscores, rbboxes = process_image_non_patch(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
