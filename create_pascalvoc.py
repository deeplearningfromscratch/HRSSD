from __future__ import absolute_import, division, print_function

import os
from lxml import etree

import tensorflow as tf

from dataset_util import read_examples_list, recursive_parse_xml_to_dict, int64_feature, bytes_feature, float_feature


def _open_image(path):
    with open(path, 'rb') as f:
        return f.read()


def _open_xml(path):
    with open(path, 'r') as f:
        return f.read()


ANNOTATION_DIR = 'Annotations/'
IMAGE_DIR = 'JPEGImages/'

YEARS = ['VOC2007', 'VOC2012', 'VOC07+12']
SETS = ['train', 'val', 'trainval', 'test']

dataset_dir = 'assets/PASCAL_VOC/'
# label_path = './furiosa-det/datasets/labels/pascalvoc_labels.yaml'
year = 'VOC2007'
set = 'test'
output_dir = 'assets/PASCAL_VOC/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

LABEL_MAP = {'aeroplane': 1,
             'bicycle': 2,
             'bird': 3,
             'boat': 4,
             'bottle': 5,
             'bus': 6,
             'car': 7,
             'cat': 8,
             'chair': 9,
             'cow': 10,
             'diningtable': 11,
             'dog': 12,
             'horse': 13,
             'motorbike': 14,
             'person': 15,
             'pottedplant': 16,
             'sheep': 17,
             'sofa': 18,
             'train': 19,
             'tvmonitor': 20}

ignore_difficult_instances = False


def serialize_example(image_file, annotation_file):
    '''
    :param image_file: image file path
    :param annotation_file: annotation file path
    :return:
    '''

    if isinstance(image_file, tf.Tensor):
        image_file = image_file.numpy()

    if isinstance(annotation_file, tf.Tensor):
        annotation_file = annotation_file.numpy()

    image_data = _open_image(image_file)
    height, width, depth = tf.image.decode_jpeg(image_data).shape

    annotation = _open_xml(annotation_file)
    root = etree.fromstring(annotation)
    data = recursive_parse_xml_to_dict(root)['annotation']

    labels = []
    labels_text = []

    ymin = []
    xmin = []
    ymax = []
    xmax = []

    difficult = []
    truncated = []

    if 'object' in data:
        for obj in data['object']:
            label = obj['name']

            if LABEL_MAP[label] is None:
                continue
            else:
                labels.append(int(LABEL_MAP[label]))
                labels_text.append(label.encode('utf-8'))

            is_difficult = bool(int(obj['difficult']))

            if ignore_difficult_instances and is_difficult:
                continue

            difficult.append(int(is_difficult))
            truncated.append(int(obj['truncated']))

            bbox = obj['bndbox']
            ymin.append(float(bbox['ymin']) / height)
            xmin.append(float(bbox['xmin']) / width)
            ymax.append(float(bbox['ymax']) / height)
            xmax.append(float(bbox['xmax']) / width)

    feature = {
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/channels': int64_feature(depth),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature('jpeg'.encode('utf-8')),
        'image/filename': bytes_feature(data['filename'].encode('utf-8')),
        'image/encoded': bytes_feature(image_data)
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(image_file, annotation_file):
    '''
    :param image_file: image file path
    :param annotation_file: annotation file path
    :return:
    '''
    tf_string = tf.py_function(
        serialize_example,
        (image_file, annotation_file),  # pass these args to the above function.
        tf.string)  # the return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar


if set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
elif year not in YEARS:
    raise ValueError('year must be in : {}'.format(YEARS))

YEARS = ['VOC2007', 'VOC2012']
if year != 'VOC07+12':
    YEARS = [year]

image_files = []
annotation_files = []
for yr in YEARS:
    dataset_root = os.path.join(dataset_dir, yr)
    image_root = os.path.join(dataset_root, IMAGE_DIR)
    annotation_root = os.path.join(dataset_root, ANNOTATION_DIR)

    if not tf.io.gfile.exists(image_root):
        raise Exception("{} does not exist".format(image_root))
    elif not tf.io.gfile.exists(annotation_root):
        raise Exception("{} does not exist".format(annotation_root))
    else:
        pass

    print('Reading from PASCAL %s %s dataset.' % (yr, set))
    examples_path = os.path.join(dataset_root, 'ImageSets', 'Main', 'aeroplane_' + set + '.txt')
    examples = read_examples_list(examples_path)
    print('%d files.' % (len(examples)))
    for ex in examples:
        image_files.append(os.path.join(image_root, ex + '.jpg'))
        annotation_files.append(os.path.join(annotation_root, ex + '.xml'))

print('Total %d files read.' % (len(image_files)))

dataset = tf.data.Dataset.from_tensor_slices((image_files, annotation_files))
serialized__dataset = dataset.map(tf_serialize_example)

tfrecord_name = 'PASCAL_%s_%s.tfrecord' % (year, set)
output_path = os.path.join(output_dir, tfrecord_name)
writer = tf.data.experimental.TFRecordWriter(output_path)
print('Writing %s ...' % (tfrecord_name))
writer.write(serialized__dataset)
print('Done!')