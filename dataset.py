import collections

import tensorflow as tf

DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor', ['splits_to_sizes', 'num_classes', 'ignore_label'])

_ADE20K_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 20210,
        'validation': 2000,
    },
    num_classes=151,
    ignore_label=0,
)

_DUMMY_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'default': 10},
    num_classes=10,
    ignore_label=0,
)

_DATASET_LIST = {
    'ade20k': _ADE20K_INFORMATION,
    'dummy': _DUMMY_INFORMATION,
}


class DatasetBuilder(object):
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
        'image/encoded':
            tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/filename':
            tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/format':
            tf.io.FixedLenFeature([], tf.string, default_value=b'jpeg'),
        'image/height':
            tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/width':
            tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/segmentation/class/format':
            tf.io.FixedLenFeature([], tf.string, default_value=b'png'),
    }

    def _parse_function(example_proto):
      return tf.io.parse_single_example(example_proto, feature_description)

    dataset = dataset.map(_parse_function,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == '__main__':
  print("abc")
  dataset = DatasetBuilder('dummy', 'default', '/tmp/dummy.tfrecord').create()
  for item in dataset.take(1):
    print(item)