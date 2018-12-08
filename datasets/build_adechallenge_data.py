from absl import flags
import glob
import math
import os
import sys

import tensorflow as tf

from utils import dataset_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_root', '/data/ADEChallengeData2016',
                    'Dataset root folder')
flags.DEFINE_string('output_dir', 'tfrecords',
                    'Path to save converted SSTable of TensorFlow Examples.')

_NUM_SHARDS = 32


def _convert_dataset(dataset_split):
  """COnverts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train, val).
  Returns:

  Raises:
  """
  image_dir = os.path.join(FLAGS.dataset_root, 'images', dataset_split, '*.jpg')
  annotation_dir = os.path.join(
      FLAGS.dataset_root, 'annotations', dataset_split, '*.png')
  image_list = sorted(glob.glob(image_dir))
  annotation_list = sorted(glob.glob(annotation_dir))

  num_samples = len(image_list)
  assert len(annotation_list) == num_samples, \
      "Annotation and image file does not match!"
  num_per_shard = int(math.ceil(num_samples / _NUM_SHARDS))

  image_reader = dataset_utils.ImageReader('jpg', channels=3)
  label_reader = dataset_utils.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    shard_filename = '%s-%05d-of-%05d.tfrecord' % (
        dataset_split, shard_id, _NUM_SHARDS)
    output_filename = os.path.join(FLAGS.dataset_root,
                                   FLAGS.output_dir,
                                   shard_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_samples)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_samples, shard_id))
        sys.stdout.flush()
        # Read the image.
        image_name = os.path.basename(image_list[i])
        label_name = os.path.basename(annotation_list[i])
        assert image_name[:-4] == label_name[:-4], \
            "image: {} and annotation: {} does not match!".format(
                image_name, label_name)
        image_data = tf.gfile.GFile(image_list[i], 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_data = tf.gfile.GFile(annotation_list[i], 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        filename = os.path.basename(image_name)
        example = dataset_utils.image_seg_to_tfexample(
            image_data,
            'jpg'.encode('utf-8'),
            filename.encode('utf-8'),
            height,
            width,
            seg_data,
            'png'.encode('utf-8'))
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  for dataset_split in ['training', 'validation']:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()