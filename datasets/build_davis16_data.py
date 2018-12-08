import absl
import math
import os
import sys

import tensorflow as tf

from utils import dataset_utils


FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string('davis16_root',
                         '/data/DAVIS',
                         'Davis16 dataset root folder.')

absl.flags.DEFINE_string(
    'output_dir',
    'tfrecords',
    'Path to save converted SSTable of TensorFlow examples.')

_NUM_SHARDS = 16


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train, val).
  Raises:
    RuntimeError: If loaded image and label have different shape, or if the
      image file with specified postfix could not be found.
  """
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.davis16_root, 'ImageSets/480p', dataset_split + '.txt'), 'r') as f:
    samples = f.readlines()

  num_samples = len(samples)
  num_per_shard = int(math.ceil(num_samples / _NUM_SHARDS))

  image_reader = dataset_utils.ImageReader('jpg', channels=3)
  label_reader = dataset_utils.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    shard_filename = '%s-%05d-of-%05d.tfrecord' % (
        dataset_split, shard_id, _NUM_SHARDS)
    output_filename = os.path.join(FLAGS.davis16_root,
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
        strs = samples[i].strip().split(' ')
        image_file = os.path.join(FLAGS.davis16_root, strs[0][1:])
        label_file = os.path.join(FLAGS.davis16_root, strs[1][1:])
        image_data = tf.gfile.GFile(image_file, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_data = tf.gfile.GFile(label_file, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        seg_img = label_reader.decode_image(seg_data)
        seg_img = seg_img // 255
        seg_data = label_reader.encode_image(seg_img)
        # Convert to tf example.
        filename = os.path.basename(image_file)
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
  for dataset_split in ['train', 'val', 'trainval']:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()