from absl import app
from absl import flags
import math
import os
import time
import tensorflow as tf
from core import deeplab_model
from datasets import dataset_factory
from tensorflow.python.estimator import estimator
import inputs
from utils import train_utils

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_name',
    default=None,
    help='Name of the Cloud TPU for Cluster Resolvers.')

# TPU settings.
flags.DEFINE_integer(
    'num_parallel_calls', default=64,
    help=('Number of parallel threads in CPU for the input pipeline'))
flags.DEFINE_integer('num_shards', 8, 'Number of shards (TPU chips).')
flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than plain CPUs')

# Dataset settings.
flags.DEFINE_string('dataset', default='davis16',
                    help='Name of the segmentation dataset.')
flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')
flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')
flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')
flags.DEFINE_bool(
    'use_cache', default=True, help=('Enable cache for training input.'))

# Learning rate settings.
flags.DEFINE_enum('optimizer', 'adam', ['adam', 'momentum'],
                  'Choose different optimizer')
flags.DEFINE_float('base_learning_rate', 0.0001,
                   'Base learning rate for training.')
flags.DEFINE_float('decay_rate', 0.97, 'The rate of decay for learning rate.')
# 8x num_shards to exploit TPU memory chunks.
flags.DEFINE_integer('eval_batch_size', 8, 'Batch size for evaluation.')
flags.DEFINE_integer('train_batch_size', 64, 'Batch size for training.')
flags.DEFINE_integer('train_steps', 20000,
                     'The number of steps to use for training.')
flags.DEFINE_integer('steps_per_eval', 2000,
                     ('Controls how often evaluation is performed.'))
flags.DEFINE_enum('mode', 'train_and_eval', ['train', 'eval', 'train_and_eval'],
                  'Train, or eval, or interleave train & eval.')
flags.DEFINE_integer('save_checkpoints_steps', 2000,
                     'Number of steps between checkpoint saves')
flags.DEFINE_string('model_dir', None, 'Estimator model_dir')
flags.DEFINE_string('init_checkpoint', None,
                    'Location of the checkpoint for seeding '
                    'the backbone network')
flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help=(
        'Maximum seconds between checkpoints before evaluation terminates.'))
# TODO(b/111116845, b/79915673): `use_host_call` must be `True`.
flags.DEFINE_bool(
    'use_host_call', default=True,
    help=('Call host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --use_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))
flags.DEFINE_integer(
    'iterations_per_loop', default=2000,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

# Model settings.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')
flags.DEFINE_integer('output_stride', 8,
                     'The ratio of input to output spatial resolution.')
flags.DEFINE_boolean('fine_tune_batch_norm',
                     True,
                     'Fine tune the batch norm parameters or not.')
flags.DEFINE_string('model_variant', 'xception_65',
                    'Upsample logits during training.')
flags.DEFINE_integer('num_gpus', 2, 'Number of GPUs in use.')

# Preprocessing settings.
flags.DEFINE_multi_integer('train_image_size', [384, 384],
                           'Cropped image size for training')
flags.DEFINE_multi_integer('eval_image_size', [384, 384],
                           'Cropped image size for evaluation')
FLAGS = flags.FLAGS


def get_distribution_strategy(num_gpus, all_reduce_alg=None):
  """Return a DistributionStrategy for running the model.

  Args:
    num_gpus: Number of GPUs to run this model.
    all_reduce_alg: Specify which algorithm to use when performing all-reduce.
      See tf.contrib.distribute.AllReduceCrossTowerOps for available algorithms.
      If None, DistributionStrategy will choose based on device topology.

  Returns:
    tf.contrib.distribute.DistibutionStrategy object.
  """
  if num_gpus == 0:
    return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  else:
    if all_reduce_alg:
      return tf.contrib.distribute.MirroredStrategy(
          num_gpus=num_gpus,
          cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
              all_reduce_alg, num_packs=num_gpus))
    else:
      return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)


def model_fn(features, labels, mode, params):
  num_classes = params['num_classes']
  model_variant = params['model_variant']
  output_stride = 8
  if 'output_stride' in params:
    output_stride = params['output_stride']
  weight_decay = 0.0001
  if 'weight_decay' in params:
    weight_decay = params['weight_decay']
  fine_tune_batch_norm = False
  if 'fine_tune_batch_norm' in params:
    fine_tune_batch_norm = params['fine_tune_batch_norm']
  add_image_level_feature = True
  if 'add_image_level_feature' in params:
    add_image_level_feature = params['add_image_level_feature']
  atrous_rates = None
  if 'atrous_rates' in params:
    atrous_rates = params['atrous_rates']
  aspp_with_separable_conv = True
  if 'aspp_with_separable_conv' in params:
    aspp_with_separable_conv = params['aspp_with_separable_conv']
  decoder_output_stride = 4
  if 'decoder_output_stride' in params:
    decoder_output_stride = params['decoder_output_stride']
  decoder_use_separable_conv = True
  if 'decoder_use_separable_conv' in params:
    decoder_use_separable_conv = params['decoder_use_separable_conv']
  depth_multiplier = 1.0
  if 'depth_multiplier' in params:
    depth_multiplier = params['depth_multiplier']
  ignore_label = -1
  if 'ignore_label' in params:
    ignore_label = params['ignore_label']
  base_learning_rate = 0.0001
  if 'base_learning_rate' in params:
    base_learning_rate = params['base_learning_rate']
  decay_rate = 0.95
  if 'decay_rate' in params:
    decay_rate = params['decay_rate']
  use_tpu = True
  if 'use_tpu' in params:
    use_tpu = params['use_tpu']

  is_training = mode == tf.estimator.ModeKeys.TRAIN

  logits = deeplab_model.build_deeplab_model(
      features,
      is_training,
      num_classes,
      model_variant,
      output_stride=output_stride,
      weight_decay=weight_decay,
      fine_tune_batch_norm=fine_tune_batch_norm,
      add_image_level_feature=add_image_level_feature,
      aspp_with_separable_conv=aspp_with_separable_conv,
      decoder_output_stride=decoder_output_stride,
      decoder_use_separable_conv=decoder_use_separable_conv,
      atrous_rates=atrous_rates,
      image_pyramid=None,
      depth_multiplier=depth_multiplier)

  label_height = tf.shape(labels)[1]
  label_width = tf.shape(labels)[2]
  # Upsample logits
  logits = tf.image.resize_bilinear(
      logits,
      [label_height, label_width],
      align_corners=True)

  logits = tf.reshape(logits, [-1, num_classes])

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

  labels = tf.reshape(labels, [-1])

  not_ignore_mask = tf.to_float(tf.not_equal(labels, ignore_label)) * 1.0
  one_hot_labels = tf.one_hot(labels, num_classes, on_value=1.0, off_value=0.0)
  cross_entropy = tf.losses.softmax_cross_entropy(
      one_hot_labels,
      logits,
      weights=not_ignore_mask,
      scope='loss_scope')

  total_loss = tf.losses.get_total_loss()


  if mode == tf.estimator.ModeKeys.TRAIN:
    num_batches_per_epoch = params['num_batches_per_epoch']
    global_step = tf.train.get_or_create_global_step()
    current_epoch = tf.cast(global_step, tf.float32) / num_batches_per_epoch

    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step,
        decay_steps=num_batches_per_epoch,
        decay_rate=decay_rate,
        staircase=True)

    if FLAGS.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif FLAGS.optimizer == 'momentum':
      optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
          momentum=0.9, use_nesterov=True)
    else:
      raise ValueError('optimizer type is not supported.')
    if use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    last_layers = deeplab_model.get_extra_layer_scopes()
    gradient_multipliers = train_utils.get_model_gradient_multipliers(
      last_layers, 10)
    train_op = tf.contrib.layers.optimize_loss(
      loss=total_loss,
      global_step=global_step,
      learning_rate=None,
      gradient_multipliers=gradient_multipliers,
      optimizer=optimizer,
      summaries=[],
      name='')  # Preventing scope prefix on all variables.

    def host_call_fn(global_step, loss, learning_rate, current_epoch):
      """Training host call. Creates scalar summaries for training metrics.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the
      model to the `metric_fn`, provide as part of the `host_call`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `host_call`.

      Args:
        global_step: `Tensor with shape `[batch, ]` for the global_step.
        loss: `Tensor` with shape `[batch, ]` for the training loss.
        learning_rate: `Tensor` with shape `[batch, ]` for the learning_rate.
        current_epoch: `Tensor` with shape `[batch, ]` for the current_epoch.

      Returns:
        List of summary ops to run on the CPU host.
      """
      # Outfeed supports int32 but global_step is expected to be int64.
      global_step = tf.reduce_mean(global_step)
      with (tf.contrib.summary.create_file_writer(
          FLAGS.model_dir).as_default()):
        with tf.contrib.summary.always_record_summaries():
          tf.contrib.summary.scalar(
            'loss', tf.reduce_mean(loss), step=global_step)
          tf.contrib.summary.scalar(
            'learning_rate', tf.reduce_mean(learning_rate),
            step=global_step)
          tf.contrib.summary.scalar(
            'current_epoch', tf.reduce_mean(current_epoch),
            step=global_step)

          return tf.contrib.summary.all_summary_ops()

    # To log the loss, current learning rate, and epoch for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly concatenated to
    # [params['batch_size']].
    global_step_t = tf.reshape(global_step, [1])
    loss_t = tf.reshape(total_loss, [1])
    learning_rate_t = tf.reshape(learning_rate, [1])
    current_epoch_t = tf.reshape(current_epoch, [1])

    host_call = (host_call_fn,
                 [global_step_t, loss_t, learning_rate_t, current_epoch_t])
  else:
    train_op = None
    host_call = None
  miou = tf.metrics.mean_iou(labels,
                             predictions['classes'],
                             num_classes,
                             weights=not_ignore_mask)
  metrics = {'miou': miou}
  def metric_fn():
    """Create metric_fn for TPUEstimatorSpec."""
    return metrics

  tpu_metrics = (metric_fn, [])

  if use_tpu:
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        host_call=host_call,
        eval_metrics=tpu_metrics)
  else:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def main(_):
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('dataset_dir')

  if FLAGS.use_tpu:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu=[FLAGS.tpu_name],
        zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)
    tpu_grpc_url = tpu_cluster_resolver.get_master()

    config = tf.contrib.tpu.RunConfig(
        master=tpu_grpc_url,
        evaluation_master=tpu_grpc_url,
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_shards))
  else:
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Create session config based on values of inter_op_parallelism_threads and
    # intra_op_parallelism_threads. Note that we default to having
    # allow_soft_placement = True, which is required for multi-GPU and not
    # harmful for other modes.
    session_config = tf.ConfigProto(
      inter_op_parallelism_threads=0,
      intra_op_parallelism_threads=0,
      allow_soft_placement=True)

    distribution_strategy = get_distribution_strategy(
        num_gpus=FLAGS.num_gpus, all_reduce_alg=None)

    config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        session_config=session_config,
        model_dir = FLAGS.model_dir)

  num_eval_samples = dataset_factory.num_samples(FLAGS.dataset,
                                                 FLAGS.eval_split)
  num_train_samples = dataset_factory.num_samples(FLAGS.dataset,
                                                  FLAGS.train_split)
  num_batches_per_epoch = math.ceil(num_train_samples / FLAGS.train_batch_size)

  # initialize our model with all but the dense layer from pretrained resnet
  if FLAGS.init_checkpoint is not None:
    warm_start_settings = tf.estimator.WarmStartSettings(
        FLAGS.init_checkpoint,
        vars_to_warm_start='^(xception_65|MobilenetV2)')
  else:
    warm_start_settings = None

  if FLAGS.use_tpu:
    classifier = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        config=config,
        warm_start_from=warm_start_settings,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        use_tpu=True,
        params={
            'num_classes': dataset_factory.num_classes(FLAGS.dataset),
            'model_variant': FLAGS.model_variant,
            'output_stride': FLAGS.output_stride,
            'atrous_rates': FLAGS.atrous_rates,
            'base_learning_rate': FLAGS.base_learning_rate,
            'ignore_label': dataset_factory.ignore_label(FLAGS.dataset),
            'decay_rate': FLAGS.decay_rate,
            'num_batches_per_epoch': num_batches_per_epoch,
            'use_tpu': FLAGS.use_tpu})

  else:
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        warm_start_from=warm_start_settings,
        params={
            'num_classes': dataset_factory.num_classes(FLAGS.dataset),
            'model_variant': FLAGS.model_variant,
            'fine_tune_batch_norm': FLAGS.fine_tune_batch_norm,
            'output_stride': FLAGS.output_stride,
            'atrous_rates': FLAGS.atrous_rates,
            'ignore_label': dataset_factory.ignore_label(FLAGS.dataset),
            'base_learning_rate': FLAGS.base_learning_rate,
            'decay_rate': FLAGS.decay_rate,
            'num_batches_per_epoch': num_batches_per_epoch,
            'use_tpu': FLAGS.use_tpu})

  train_preprocess_config = {
    'model_variant': 'xception_65',
    'crop_height': FLAGS.train_image_size[0],
    'crop_width': FLAGS.train_image_size[1],
    'min_resize_value': None,
    'max_resize_value': None,
    'resize_factor': None,
    'min_scale_factor': 1.0,
    'max_scale_factor': 1.0,
    'scale_factor_step_size': 0.0,
    'force_valid_scale': False}
  eval_preprocess_config = {
    'model_variant': 'xception_65',
    'crop_height': FLAGS.eval_image_size[0],
    'crop_width': FLAGS.eval_image_size[1],
    'min_resize_value': None,
    'max_resize_value': None,
    'resize_factor': None,
    'min_scale_factor': 1.0,
    'max_scale_factor': 1.0,
    'scale_factor_step_size': 0.0,
    'force_valid_scale': False}

  data_train = inputs.DeeplabInput(
      is_training=True,
      name=FLAGS.dataset,
      split=FLAGS.train_split,
      dataset_dir=FLAGS.dataset_dir,
      num_parallel_calls=FLAGS.num_parallel_calls,
      cache=FLAGS.use_cache,
      batch_size=FLAGS.train_batch_size,
      preprocess_config=train_preprocess_config)
  data_eval = inputs.DeeplabInput(
      is_training=False,
      name=FLAGS.dataset,
      split=FLAGS.eval_split,
      dataset_dir=FLAGS.dataset_dir,
      num_parallel_calls=FLAGS.num_parallel_calls,
      cache=False,
      batch_size=FLAGS.eval_batch_size,
      preprocess_config=eval_preprocess_config)


  if FLAGS.mode == 'train':
    tf.logging.info('Training for %d steps (%.2f epochs in total).' %
                    (FLAGS.train_steps,
                     FLAGS.train_steps / num_batches_per_epoch))
    classifier.train(
        input_fn=data_train.input_fn,
        max_steps=FLAGS.train_steps)

  elif FLAGS.mode == 'eval':
    # Run evaluation when there's a new checkpoint
    for ckpt in tf.contrib.training.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):

      tf.logging.info('Starting to evaluate.')
      try:
        eval_results = classifier.evaluate(
            input_fn=data_eval.input_fn,
            steps=math.ceil(num_eval_samples / FLAGS.eval_batch_size)
        )
        tf.logging.info('Eval results: %s' % eval_results)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= FLAGS.train_steps:
          tf.logging.info('Evaluation finished after training step %d' %
                          current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info('Checkpoint %s no longer exists, skipping checkpoint' %
                        ckpt)
  elif FLAGS.mode == 'train_and_eval':
    """Interleaves training and evaluation."""
    # pylint: disable=protected-access
    current_step = estimator._load_global_step_from_checkpoint_dir(
      FLAGS.model_dir)
    tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                    ' step %d.' %
                    (FLAGS.train_steps,
                     FLAGS.train_steps / num_batches_per_epoch,
                     current_step))
    start_timestamp = time.time()
    while current_step < FLAGS.train_steps:
      # Train for up to steps_per_eval number of steps. At the end of training,
      # a checkpoint will be written to --model_dir.
      next_checkpoint = int(min(current_step + FLAGS.steps_per_eval,
                                FLAGS.train_steps))

      classifier.train(
          input_fn=data_train.input_fn,
          max_steps=next_checkpoint)
      current_step = next_checkpoint

      elapsed_time = int(time.time() - start_timestamp)
      tf.logging.info('Finished training up to step %d. Elapsed seconds %d.' %
                      (current_step, elapsed_time))

      tf.logging.info('Starting to evaluate.')

      eval_results = classifier.evaluate(
          input_fn=data_eval.input_fn,
          steps=math.ceil(num_eval_samples / FLAGS.eval_batch_size))
      tf.logging.info('Eval results: %s' % eval_results)
  else:
    tf.logging.error('Mode not found.')

  tf.logging.info('Done!')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
