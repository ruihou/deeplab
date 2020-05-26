import tensorflow as tf


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def segmentation_image_to_tfexample(
    image_data: str,
    annotation_data: str,
    height: int,
    width: int,
    filename: str,
    image_format: str = 'jpeg',
    annotation_format: str = 'png') -> tf.train.Example:
  """Converts one image annotation pair to tf example.
  Args:
    image_data: String of image data.
    annotation_data: String of semantic segmentation data.
    height: Image height.
    width: Image width.
    filename: Image filename.
    image_format: Image format, jpeg or png.
    annotation_format: Segmentation image format.
  Returns:
    A tf example instance.
  """
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_feature(image_data),
      'image/filename': _bytes_feature(filename.encode('utf-8')),
      'image/format': _bytes_feature(image_format.encode('utf-8')),
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/channels': _int64_feature(3),
      'image/segmentation/class/encoded': (_bytes_feature(annotation_data)),
      'image/segmentation/class/format': _bytes_feature(
          annotation_format.encode('utf-8')),
  }))