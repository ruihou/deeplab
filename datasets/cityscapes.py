from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from utils import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 2975, 'val': 500, 'test': 500}

_NUM_CLASSES = 19

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': ('A semantic segmentation label whose size matches image.'
              'Its values range from 0 (background) to num_classes.'),
}

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature((), tf.int64, default_value=0),
      'image/segmentation/class/encoded': tf.FixedLenFeature((), tf.string,
                                                             default_value=''),
      'image/segmentation/class/format': tf.FixedLenFeature((), tf.string,
                                                            default_value='png')
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(image_key='image/encoded',
                                            format_key='image/format',
                                            channels=3),
      'height': slim.tfexample_decoder.Tensor('image/height'),
      'width': slim.tfexample_decoder.Tensor('image/width'),
      'label': slim.tfexample_decoder.Image(
          image_key='image/segmentation/class/encoded',
          format_key='image/segmentation/class/encoded',
          channels=1),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_filename = os.path.join(dataset_dir, 'labels.txt')
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index + 1:]

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      ignore_label=255,
      num_classes=_NUM_CLASSES,
      multi_label=True,
      labels_to_names=labels_to_class_names)