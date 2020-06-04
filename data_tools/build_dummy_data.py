import absl.app
import math
import numpy as np
import os.path
import sys

import tensorflow as tf

from utils import dataset_utils

FLAGS = absl.app.flags.FLAGS

absl.app.flags.DEFINE_string('output_dir', '/tmp/',
                             'Path to save converted tfrecord of tf.example')
absl.app.flags.DEFINE_integer('num_shards', 1,
                              'Number of shards in output tfrecords')


def _create_dummy_dataset():
  """Create dummy dataset, and save it into tfrecord.
  """
  image_np = np.arange(224 * 224 * 3).reshape((224, 224, 3))
  annotation_np = np.arange(224 * 224).reshape((224, 224, 1))

  for shard_id in range(FLAGS.num_shards):
    if FLAGS.num_shards == 1:
      output_filename = os.path.join(FLAGS.output_dir, 'dummy.tfrecord')
    else:
      output_filename = os.path.join(
          FLAGS.output_dir,
          '%s-%05d-of-%05d.tfrecord' % ('dummy', shard_id, FLAGS.num_shards))
    with tf.io.TFRecordWriter(output_filename) as writer:
      image_name = 'image_{}'.format(shard_id)

      image_data = (image_np + shard_id) % 256
      image_data = tf.image.encode_png(image_data.astype(np.uint8))
      annotation_data = (annotation_np + shard_id) % 10
      annotation_data = tf.image.encode_png(annotation_data.astype(np.uint8))
      height = 224
      width = 224

      example = dataset_utils.segmentation_image_to_tfexample(
          image_data, annotation_data, height, width, image_name,
          image_format='png')
      writer.write(example.SerializeToString())


def main(unused_argv):
  _create_dummy_dataset()


if __name__ == '__main__':
  absl.app.run(main)