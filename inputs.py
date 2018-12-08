import abc
import functools
import os
import tensorflow as tf
from core import input_preprocess
from datasets import dataset_factory

slim = tf.contrib.slim
_FILE_PATTERN = '%s*.tfrecord'


class DeeplabTFExampleInput(object):
  """Base class for Deeplab semantic segmentation input_fn generator.

  Args:
    is_training: 'bool' for whether the input is for training
    num_cores: 'int' for the number of TPU cores
    name: 'str' for name of the dataset.
    split: train/val split name.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, is_training, name, split, num_cores=8,
               batch_size=None, preprocess_config={}):
    self.preprocessing_fn = self.get_preprocess_fn(preprocess_config)
    self.is_training = is_training
    self.name = name
    self.split = split
    self.num_cores = num_cores
    self.batch_size = batch_size

  def get_preprocess_fn(self, configs):
    model_variant = 'xception_65'
    if 'model_variant' in configs:
      model_variant = configs['model_variant']
    crop_height = None
    if 'crop_height' in configs:
      crop_height = configs['crop_height']
    crop_width = None
    if 'crop_width' in configs:
      crop_width = configs['crop_width']
    min_resize_value = None
    if 'min_resize_value' in configs:
      min_resize_value = configs['min_resize_value']
    max_resize_value = None
    if 'max_resize_value' in configs:
      max_resize_value = configs['max_resize_value']
    resize_factor = None
    if 'resize_vactor' in configs:
      resize_factor = configs['resize_factor']
    min_scale_factor = 1.0
    if 'min_scale_factor' in configs:
      min_scale_factor = configs['min_scale_factor']
    max_scale_factor = 1.0
    if 'max_scale_factor' in configs:
      max_scale_factor = configs['max_scale_factor']
    scale_factor_step_size = 0.0
    if 'scale_factor_step_size' in configs:
      scale_factor_step_size = configs['scale_factor_step_size']
    force_valid_scale = False
    if 'force_valid_scale' in configs:
      force_valid_scale = configs['force_valid_scale']
    return functools.partial(input_preprocess.preprocess_image_and_label,
                             model_variant=model_variant,
                             crop_height=crop_height,
                             crop_width=crop_width,
                             min_resize_value=min_resize_value,
                             max_resize_value=max_resize_value,
                             resize_factor=resize_factor,
                             min_scale_factor=min_scale_factor,
                             max_scale_factor=max_scale_factor,
                             scale_factor_step_size=scale_factor_step_size,
                             force_valid_scale=force_valid_scale)

  def set_shapes(self, batch_size, images, labels):
    """Statically set the batch)size dimension."""
    images.set_shape(images.get_shape().merge_with(
        tf.TensorShape([batch_size, None, None, 3])))
    labels.set_shape(labels.get_shape().merge_with(
        tf.TensorShape([batch_size, None, None, 1])))

    return images, labels

  def dataset_parser(self, value):
    """Parses an image and its label from a serialized segmentation TFExample.

    Args:
      value: serialized string containing an segmentation TFExample.


    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string,
          default_value=dataset_factory.image_format(self.name)),
      'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature((), tf.int64, default_value=0),
      'image/segmentation/class/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/segmentation/class/format': tf.FixedLenFeature((), tf.string,
           default_value=dataset_factory.label_format(self.name))
    }

    parsed = tf.parse_single_example(value, keys_to_features)

    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    label_bytes = tf.reshape(
        parsed['image/segmentation/class/encoded'], shape=[])
    image = tf.image.decode_image(image_bytes, channels=3)
    label = tf.image.decode_image(label_bytes, channels=1)

    [image, label] = self.preprocessing_fn(
        image, label, ignore_label=dataset_factory.ignore_label(self.name),
        is_training=self.is_training)
    return image, label

  @abc.abstractmethod
  def make_source_dataset(self, index, num_hosts):
    """Makes dataset of serialized TFExamples.
    The returned dataset will contain `tf.string` tensors, but these strings are
    serialized `TFExample` records that will be parsed by `dataset_parser`.
    If self.is_training, the dataset should be infinite.
    Args:
      index: current host index.
      num_hosts: total number of hosts.
    Returns:
      A `tf.data.Dataset` object.
    """
    return

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.
    Returns:
      A `tf.data.Dataset` object.
    """
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.contrib.tpu.RunConfig for details.
    if 'batch_size' in params:
      batch_size = params['batch_size']
    else:
      batch_size = self.batch_size

    # TODO(dehao): Replace the following with params['context'].current_host
    if 'context' in params:
      current_host = params['context'].current_input_fn_deployment()[1]
      num_hosts = params['context'].num_hosts
    else:
      current_host = 0
      num_hosts = 1

    dataset = self.make_source_dataset(current_host, num_hosts)

    # Use the fused map-and-batch operation.
    #
    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
    # batches without dropping any training examples.
    #
    # When evaluating, `drop_remainder=True` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. As long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            self.dataset_parser, batch_size=batch_size,
            num_parallel_batches=self.num_cores, drop_remainder=True))

    # Assign static batch size dimension
    dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset


class DeeplabInput(DeeplabTFExampleInput):
  """Generates deeplab input_fn from a series of TFRecord files.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across files.
  """

  def __init__(self,
               is_training,
               name,
               split,
               dataset_dir,
               num_parallel_calls=64,
               cache=False,
               batch_size=None,
               preprocess_config={}):
    """Create an input from TFRecord files.

    Args:
      is_training: 'bool' for whether the input is for training
      name: 'str' for name of the dataset.
      split: train/test split name.
      dataset_dir: 'str' for the directory of the training and validation data;
          if 'null' (the literal string 'null') of implicitly False
          then construct a null pipeline, consisting of empty images
          and blank labels.
      num_parallel_calls: concurrency level to use when reading data from disk.
      cache: if true, fill the dataset by repeating from its cache
    """
    super(DeeplabInput, self).__init__(
        is_training=is_training,
        name=name,
        split=split,
        batch_size=batch_size,
        preprocess_config=preprocess_config)
    self.dataset_dir = dataset_dir,
    self.num_parallel_calls = num_parallel_calls
    self.cache = cache

  def dataset_parser(self, value):
    """See base class."""
    if not self.dataset_dir:
      return value, tf.constant(0, tf.int32)
    return super(DeeplabInput, self).dataset_parser(value)

  def make_source_dataset(self, index, num_hosts):
    """See base class"""
    if not self.dataset_dir:
      raise ValueError('Undefined dataset_dir')

    # Shuffle the filenames to ensure better randomization.
    file_pattern = os.path.join(self.dataset_dir[0], _FILE_PATTERN % self.split)

    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    dataset = dataset.shard(num_hosts, index)

    if self.is_training and not self.cache:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 24 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(
        fetch_dataset, cycle_length=self.num_parallel_calls, sloppy=True))

    if self.cache:
      dataset = dataset.cache().apply(
          tf.contrib.data.shuffle_and_repeat(1024 * 16))
    else:
      dataset = dataset.shuffle(1024)
    return dataset