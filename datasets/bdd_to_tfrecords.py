from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from datasets.bdd_common import label_map
import os
import json
import tensorflow as tf
import sys
import random

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = '../assets/BDD/TEST_objects.json'
DIRECTORY_IMAGES = '../assets/BDD/TEST_images.json'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200
data_folder = '../assets/BDD'
split = 'TEST'


def _process_image(img, obj):
    filename = img
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    shape = [720, 1280, 3]

    bboxes = []

    for box in obj['boxes']:
        bboxes.append((float(box[1] / shape[0]),
                       float(box[0] / shape[1]),
                       float(box[3] / shape[0]),
                       float(box[2] / shape[1])
                       ))

    labels = obj['labels']

    return image_data, shape, bboxes, labels


def _convert_to_example(image_data, labels, bboxes, shape):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(image_path, objects, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, = _process_image(image_path, objects)
    example = _convert_to_example(image_data, labels, bboxes, shape)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, split, name='bdd_train'):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames, and shuffling.

    with open(os.path.join(dataset_dir, split + '_images.json'), 'r') as j:
        images = json.load(j)
    with open(os.path.join(dataset_dir, split + '_objects.json'), 'r') as j:
        objects = json.load(j)

    assert len(images) == len(objects)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(images):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(images) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(images)))
                sys.stdout.flush()

                image_path = images[i]
                obj = objects[i]
                _add_to_tfrecord(image_path, obj, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the BDD dataset!')


dataset_dir = '../assets/BDD'
run(dataset_dir, dataset_dir, 'TEST', 'bdd_test')