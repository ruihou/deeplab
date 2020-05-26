import absl.app
import math
import os.path
import sys

import tensorflow as tf

from utils import dataset_utils

FLAGS = absl.app.flags.FLAGS

absl.app.flags.DEFINE_string('dataset_dir', '',
                             'Folder containing raw dataset')
absl.app.flags.DEFINE_string('output_dir', '',
                             'Path to save converted tfrecord of tf.example')
absl.app.flags.DEFINE_integer('num_shards', 4,
                              'Number of shards in output tfrecords')


def _convert_dataset(split: str):
  """Converts the ADEChallenge2016 dataset into tfrecord.
  Args:
    split: Dataset split.

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  image_dir = os.path.join(FLAGS.dataset_dir, 'images', split)
  annotation_dir = os.path.join(FLAGS.dataset_dir, 'annotations', split)

  image_names = tf.io.gfile.glob(os.path.join(image_dir, '*.jpg'))

  num_images = len(image_names)
  num_per_shard = int(math.ceil(num_images / FLAGS.num_shards))

  for shard_id in range(FLAGS.num_shards):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (split, shard_id, FLAGS.num_shards))
    with tf.io.TFRecordWriter(output_filename) as writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
          i + 1, num_images, shard_id))
        sys.stdout.flush()

        image_name = image_names[i]
        filename = os.path.basename(image_name).split('.')[0]
        annotation_name = os.path.join(annotation_dir, filename + '.png')

        image_data = tf.io.gfile.GFile(image_name, 'rb').read()
        height, width, _ = tf.io.decode_jpeg(image_data, channels=3).shape

        annotation_data = tf.io.gfile.GFile(annotation_name, 'rb').read()
        annotation_height, annotation_width, _ = tf.io.decode_png(
            annotation_data, channels=1).shape
        if height != annotation_height or width != annotation_width:
          raise RuntimeError('Shape mismatched between image and annotation.')

        example = dataset_utils.segmentation_image_to_tfexample(
            image_data, annotation_data, height, width, filename)
        writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def main(unused_argv):
  _convert_dataset('training')
  _convert_dataset('validation')


if __name__ == '__main__':
  absl.app.run(main)