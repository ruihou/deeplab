import functools
import os
import tensorflow as tf

from protos import input_reader_pb2
from utils import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s-*.tfrecord'

SPLITS_TO_SIZES = {'train': 2079, 'test': 1376}

_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': ('A semantic segmentation label whose size matches image.'
              'Its values range from 0 (background) to num_classes.'),
}

def get_split(
    split_name, dataset_dir, file_pattern=None, config=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    config: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if not config:
    config = input_reader_pb2.InputReader()

  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
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

  def process_fn(value):
    """Sets up tf graph that decodes input data."""
    processed_tensors = decoder.decode(value)
    keys = decoder.list_items()
    return dict(zip(keys, processed_tensors))

  dataset = dataset_utils.read_dataset(
    functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
    file_pattern, config)
  if config.sample_1_of_n_examples > 1:
   dataset = dataset.shard(config.sample_1_of_n_examples, 0)

  dataset = dataset.map(
    process_fn,
    num_parallel_calls=config.batch_size)

  return dataset
