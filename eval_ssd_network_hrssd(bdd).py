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

slim = tf.contrib.slim

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
    'batch_size', 8, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', 'checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_string(
    'dataset_name', 'bdd', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', 'assets/BDD/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'model_name', 'ssd_512_vgg', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')
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
        ssd_net_base = ssd_class(ssd_class.default_params)
        # base_shape = (300, 300)
        base_shape = (512, 512)
        ssd_anchors_base = ssd_net_base.anchors(base_shape)

        # ssd_shape = (1024, 1024)
        # feat_shapes = [(128, 128), (64, 64), (32, 32)]#, (16, 16) ]#(64, 64), (32, 32), (16, 16), (14, 14), (12, 12)]
        ssd_shape = (720, 1280)
        feat_shapes = [(90, 160), (45, 80), (23, 40)]  # , (16, 16) ]#(64, 64), (32, 32), (16, 16), (14, 14), (12, 12)]
        # feat_shapes = [(128, 128), (64, 64), (32, 32), (16, 16), (14, 14), (12, 12)]
        feat_layers = ['block4',
                       'block7',
                       'block8',]
                       # 'block9']
                       # 'block9',]
                       # 'block10',
                       # 'block11']
        # ['block4', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12']
        ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes, img_shape=ssd_shape,
                                                       feat_shapes=feat_shapes, feat_layers=feat_layers)
        ssd_net = ssd_class(ssd_params)
        ssd_anchors = ssd_net.anchors(ssd_shape)

        # Evaluation shape and associated anchors: eval_image_size
        # ssd_shape = ssd_net.params.img_shape



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
            # if FLAGS.remove_difficult:
            #     [gdifficults] = provider.get(['object/difficult'])
            # else:
            #     gdifficults = tf.zeros(tf.shape(glabels), dtype=tf.int64)
            gdifficults = tf.zeros(tf.shape(glabels), dtype=tf.int64)

            # Pre-processing image, labels and bboxes.
            image, _, _, _ = \
                image_preprocessing_fn(image, None, None,
                                       out_shape=ssd_shape,
                                       data_format=DATA_FORMAT,
                                       resize=FLAGS.eval_resize,
                                       difficults=None)

            # image, glabels, gbboxes, gbbox_img = \
            #     image_preprocessing_fn(image, glabels, gbboxes,
            #                            out_shape=base_shape,
            #                            data_format=DATA_FORMAT,
            #                            resize=FLAGS.eval_resize,
            #                            difficults=None)

            image_b, glabels_b, gbboxes_b, gbbox_img_b = \
                image_preprocessing_fn(image, glabels, gbboxes,
                                       out_shape=base_shape,
                                       data_format=DATA_FORMAT,
                                       resize=FLAGS.eval_resize,
                                       difficults=None)

            # Encode groundtruth labels and bboxes.
            # gclasses_b, glocalisations_b, gscores_b = \
            #     ssd_net.bboxes_encode(glabels_b, gbboxes_b, ssd_anchors_base)
            batch_shape = [1] * 5

            # gclasses += gclasses_b
            # glocalisations += glocalisations_b
            # gscores += gscores_b

            # gclasses = gclasses_b
            # glocalisations = glocalisations_b
            # gscores = gscores_b

            # glabels = glabels_b
            # gbboxes = gbboxes_b

            # Evaluation batch.
            r = tf.train.batch(
                tf_utils.reshape_list([image, image_b, glabels, gbboxes, gdifficults]),
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size,
                dynamic_pad=True)
            (b_image, b_image_b, b_glabels, b_gbboxes, b_gdifficults) = tf_utils.reshape_list(r, batch_shape)
            # r = tf.train.batch(
            #     tf_utils.reshape_list([image_b]),
            #     batch_size=FLAGS.batch_size,
            #     num_threads=FLAGS.num_preprocessing_threads,
            #     capacity=5 * FLAGS.batch_size,
            #     dynamic_pad=True)
            # (b_image_b) = tf_utils.reshape_list(r, batch_shape)

        # =================================================================== #
        # SSD Network + Ouputs decoding.
        # =================================================================== #
        dict_metrics = {}

        arg_scope = ssd_net.arg_scope(data_format=DATA_FORMAT)
        with slim.arg_scope(arg_scope):
            predictions, localisations, logits, end_points = \
                ssd_net.net(b_image, is_training=False, reuse=tf.AUTO_REUSE)

        with slim.arg_scope(arg_scope):
            predictions_b, localisations_b, logits, end_points = \
                ssd_net_base.net(b_image_b, is_training=False, reuse=tf.AUTO_REUSE)

        predictions += predictions_b

        # Add losses functions.
        # ssd_net.losses(logits, localisations,
        #                b_gclasses, b_glocalisations, b_gscores)

        # Performing post-processing on CPU: loop-intensive, usually more efficient.
        with tf.device('/device:CPU:0'):
            # Detected objects from SSD output.
            localisations = ssd_net.bboxes_decode(localisations, ssd_anchors)
            localisations_b = ssd_net_base.bboxes_decode(localisations_b, ssd_anchors_base)
            localisations += localisations_b

            rscores, rbboxes = \
                ssd_net.detected_bboxes(predictions, localisations,
                                        select_threshold=FLAGS.select_threshold,
                                        nms_threshold=FLAGS.nms_threshold,
                                        clipping_bbox=None,
                                        top_k=FLAGS.select_top_k,
                                        keep_top_k=FLAGS.keep_top_k)

            for cls in rscores.keys():
            #     # rscores[cls] = tf.Print(rscores[cls], [rscores[cls]], 'before %d scores' % cls,
            #     #                         summarize=rscores[cls].shape[1])
            #     # rbboxes[cls] = tf.Print(rbboxes[cls], [rbboxes[cls]], 'before %d bboxes' % cls,
            #     #                         summarize=rbboxes[cls].shape[1])
                labels = [6, 15, 2, 14, 7, 19, 15]
            #     # if cls not in labels:
            #     #     rscores[cls] = tf.zeros_like(rscores[cls])
            #     #     rbboxes[cls] = tf.zeros_like(rbboxes[cls])
            #
                if cls not in labels:
                    rscores[cls] = tf.zeros_like(rscores[cls])
                    rbboxes[cls] = tf.zeros_like(rbboxes[cls])
            #         # rscores[cls] = tf.Print(rscores[cls], [rscores[cls]], 'after %d scores' % cls,
            #         #                         summarize=rscores[cls].shape[1])
            #         # rbboxes[cls] = tf.Print(rbboxes[cls], [rbboxes[cls]], 'after %d bboxes' % cls,
            #         #                         summarize=rbboxes[cls].shape[1])
            #
            #     # # if cls in labels:
            #     #     rscores[cls] = tf.Print(rscores[cls], [rscores[cls]], '%d scores' % cls,
            #     #                             summarize=rscores[cls].shape[1])

            # Compute TP and FP statistics.
            num_gbboxes, tp, fp, rscores = \
                tfe.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                                          b_glabels, b_gbboxes, b_gdifficults,
                                          matching_threshold=FLAGS.matching_threshold)

        # Variables to restore: moving avg. or normal weights.
        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        # =================================================================== #
        # Evaluation metrics.
        # =================================================================== #
        with tf.device('/device:CPU:0'):
            dict_metrics = {}
            # First add all losses.
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)
            # Extra losses as well.
            for loss in tf.get_collection('EXTRA_LOSSES'):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)

            # Add metrics to summaries and Print on screen.
            for name, metric in dict_metrics.items():
                # summary_name = 'eval/%s' % name
                summary_name = name
                op = tf.summary.scalar(summary_name, metric[0], collections=[])
                # op = tf.Print(op, [metric[0]], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

            # FP and TP metrics.
            tp_fp_metric = tfe.streaming_tp_fp_arrays(num_gbboxes, tp, fp, rscores)
            for c in tp_fp_metric[0].keys():
                dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c],
                                                tp_fp_metric[1][c])

            # Add to summaries precision/recall values.
            aps_voc07 = {}
            aps_voc12 = {}
            for c in tp_fp_metric[0].keys():
                # Precison and recall values.
                prec, rec = tfe.precision_recall(*tp_fp_metric[0][c])

                # Average precision VOC07.
                v = tfe.average_precision_voc07(prec, rec)
                summary_name = 'AP_VOC07/%s' % c
                op = tf.summary.scalar(summary_name, v, collections=[])
                # op = tf.Print(op, [v], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                aps_voc07[c] = v

                # Average precision VOC12.
                v = tfe.average_precision_voc12(prec, rec)
                summary_name = 'AP_VOC12/%s' % c
                op = tf.summary.scalar(summary_name, v, collections=[])
                # op = tf.Print(op, [v], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                aps_voc12[c] = v

            # Mean average precision VOC07.
            summary_name = 'AP_VOC07/mAP'
            mAP = tf.add_n(list(aps_voc07.values())) / len(aps_voc07)
            op = tf.summary.scalar(summary_name, mAP, collections=[])
            op = tf.Print(op, [mAP], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

            # Mean average precision VOC12.
            summary_name = 'AP_VOC12/mAP'
            mAP = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
            op = tf.summary.scalar(summary_name, mAP, collections=[])
            op = tf.Print(op, [mAP], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # for i, v in enumerate(l_precisions):
        #     summary_name = 'eval/precision_at_recall_%.2f' % LIST_RECALLS[i]
        #     op = tf.summary.scalar(summary_name, v, collections=[])
        #     op = tf.Print(op, [v], summary_name)
        #     tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # Split into values and updates ops.
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(dict_metrics)

        # =================================================================== #
        # Evaluation loop.
        # =================================================================== #
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        # Number of batches...
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        def flatten(x):
            result = []
            for el in x:
                if isinstance(el, tuple):
                    result.extend(flatten(el))
                else:
                    result.append(el)
            return result

        if not FLAGS.wait_for_checkpoints:
            if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            else:
                checkpoint_path = FLAGS.checkpoint_path
            tf.logging.info('Evaluating %s' % checkpoint_path)

            def flatten(seq):
                l = []

                for elt in seq:
                    t = type(elt)
                    if t is tuple or t is list:
                        for elt2 in flatten(elt):
                            l.append(elt2)
                    else:
                        l.append(elt)
                return l

            # Standard evaluation loop.
            start = time.time()
            slim.evaluation.evaluate_once(
                master=FLAGS.master,
                checkpoint_path=checkpoint_path,
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
                eval_op=flatten(list(names_to_updates.values())),
                variables_to_restore=variables_to_restore,
                session_config=config)
            # Log time spent.
            elapsed = time.time()
            elapsed = elapsed - start
            print('Time spent : %.3f seconds.' % elapsed)
            print('Time spent per BATCH: %.3f seconds.' % (elapsed / num_batches))

        else:
            checkpoint_path = FLAGS.checkpoint_path
            tf.logging.info('Evaluating %s' % checkpoint_path)

            # Waiting loop.
            slim.evaluation.evaluation_loop(
                master=FLAGS.master,
                checkpoint_dir=checkpoint_path,
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
                eval_op=flatten(list(names_to_updates.values())),
                variables_to_restore=variables_to_restore,
                eval_interval_secs=60,
                max_number_of_evaluations=np.inf,
                session_config=config,
                timeout=None)


if __name__ == '__main__':
    tf.app.run()
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