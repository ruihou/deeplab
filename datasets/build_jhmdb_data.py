from absl import flags
import math
import numpy as np
import os
import scipy.io as sio
import sys
import tensorflow as tf
from utils import dataset_utils

flags.DEFINE_string('dataset_root', '/data/jhmdb',
                    'Dataset root folder')
flags.DEFINE_string('output_dir', 'tfrecords',
                    'Path to save converted SSTable of TensorFlow Examples.')

FLAGS = flags.FLAGS

_LABEL_LIST = ['brush_hair', 'catch', 'clap', 'climb_stairs', 'golf',
               'jump', 'kick_ball', 'pick', 'pour', 'pullup',
               'push', 'run', 'shoot_ball', 'shoot_bow', 'shoot_gun',
               'sit', 'stand', 'swing_baseball', 'throw', 'walk',
               'wave']
_NUM_SHARDS = 16


def load_train_val_split(split_name):
  train_set = []
  test_set = []
  for label in _LABEL_LIST:

    split_file = os.path.join(FLAGS.dataset_root, 'splits',
                              '{}_test_{}.txt'.format(label, split_name))
    with open(split_file, 'r') as f:
      lines = f.readlines()
      for line in lines:
        [video_name, split] = line.strip().split(' ')
        split = int(split)
        if split == 1:
          train_set.append({'name': video_name[:-4], 'label': label})
        elif split == 2:
          test_set.append({'name': video_name[:-4], 'label': label})
        else:
          raise ValueError("Wrong split: {}____{}".format(line, split))

  return train_set, test_set


def load_annotation(video_set):
  total_frames = 0
  for video in video_set:
    annotation_file = os.path.join(
        FLAGS.dataset_root, 'puppet_mask', video['label'], video['name'],
        'puppet_mask.mat')
    mask = sio.loadmat(annotation_file)['part_mask']
    video['mask'] = mask
    video['num_frames'] = mask.shape[2]
    total_frames += mask.shape[2]
  return video_set, total_frames


def write_to_tfrecords(name, video_set):
  num_samples = len(video_set)
  videos_per_shard = int(math.ceil(num_samples / _NUM_SHARDS))
  image_reader = dataset_utils.ImageReader('png', channels=3)
  label_reader = dataset_utils.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    shard_filename = '%s-%05d-of-%05d.tfrecord' % (
        name, shard_id, _NUM_SHARDS)
    output_filename = os.path.join(FLAGS.dataset_root,
                                   FLAGS.output_dir,
                                   shard_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * videos_per_shard
      end_idx = min((shard_id + 1) * videos_per_shard, num_samples)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting video %d/%d shard %d' % (
            i + 1, num_samples, shard_id))
        sys.stdout.flush()
        # Read the image.
        for j in range(video_set[i]['num_frames']):
          image_name = os.path.join(FLAGS.dataset_root,
                                    'Rename_Images',
                                    video_set[i]['label'],
                                    video_set[i]['name'],
                                    '{:05d}.png'.format(j + 1))
          image_data = tf.gfile.GFile(image_name, 'rb').read()
          height, width = image_reader.read_image_dims(image_data)
          curr_mask = video_set[i]['mask'][:,:,j]
          curr_mask = np.expand_dims(curr_mask, axis=2)
          seg_data = label_reader.encode_image(curr_mask)
          if height != curr_mask.shape[0] or width != curr_mask.shape[1]:
            raise RuntimeError('Shape mismatched between image and label.')
          # Convert to tf example.
          filename = os.path.basename(image_name)
          example = dataset_utils.image_seg_to_tfexample(
              image_data,
              'png'.encode('utf-8'),
              filename.encode('utf-8'),
              height,
              width,
              seg_data,
              'png'.encode('utf-8'))
          tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def _convert_dataset(split_name):
  train_set, test_set = load_train_val_split(split_name)

  train_set, num_frames = load_annotation(train_set)
  #write_to_tfrecords(split_name + '_train', train_set)
  print('{} from train set'.format(num_frames))

  test_set, num_frames = load_annotation(test_set)
  #write_to_tfrecords(split_name + '_val', test_set)
  print('{} from test set'.format(num_frames))


def main(unused_argv):
  for dataset_split in ['split1', 'split2', 'split3']:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()