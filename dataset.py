import collections

import tensorflow as tf

DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor', ['splits_to_sizes', 'num_classes', 'ignore_label'])

_DATASET_LIST = {
    'ade20k': _ADE20K_INFORMATION,
}

_ADE20K_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 20210,
        'validation': 2000,
    },
    num_classes=151,
    ignore_label=0,
)


class Dataset(object):
  def __init__(self,
               name: str,
               split: str,
               dataset_dir: str):
    """Initializes the dataset.
    Args:
      name: Name of the dataset.
      split: Name of the split (e.g. train/val).
      dataset_dir: Path to the dataset tfrecords.

    Raises:
      ValueError:
    """
    if name not in _DATASET_LIST:
      raise ValueError('The specified dataset is not supported yet.')

    splits_to_sizes = _DATASET_LIST[name].splits_to_sizes
    if split not in splits_to_sizes:
      raise ValueError('data split name {} is not recognized'.format(split))
    self.num_samples = _DATASET_LIST[name].splits_to_sizes[split]
    self.num_classes = _DATASET_LIST[name].num_classes
    self.ignore_label = _DATASET_LIST[name].ignore_label
    self.dataset_dir = dataset_dir

  def create(self):
    file_dataset = tf.data.Dataset.list_files(self.dataset_dir)
    dataset = tf.data.TFRecordDataset(file_dataset)

    feature_description = {
      'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }

    def _parse_function(example_proto):
      return tf.io.parse_single_example(example_proto, feature_description)

    dataset.map(_parse_function,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset