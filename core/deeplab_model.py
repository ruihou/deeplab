import tensorflow as tf
from core import feature_extractor

slim = tf.contrib.slim

LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'


def get_extra_layer_scopes(last_layers_contain_logits_only=False):
  """Gets the scopes for extra layers.

  Args:
    last_layers_contain_logits_only: Boolean, True if only consider logits as
    the last layer (i.e., exclude ASPP module, decoder module and so on)

  Returns:
    A list of scopes for extra layers.
  """
  if last_layers_contain_logits_only:
    return [LOGITS_SCOPE_NAME]
  else:
    return [
        LOGITS_SCOPE_NAME,
        IMAGE_POOLING_SCOPE,
        ASPP_SCOPE,
        CONCAT_PROJECTION_SCOPE,
        DECODER_SCOPE,
    ]

def scale_dimension(dim, scale):
  """Scales the input dimension.

  Args:
    dim: Input dimension (a scalar or a scalar Tensor).
    scale: The amount of scaling applied to the input.

  Returns:
    Scaled dimension.
  """
  if isinstance(dim, tf.Tensor):
    return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)


def build_deeplab_model(
    images,
    is_training,
    num_classes,
    model_variant,
    output_stride=8,
    weight_decay=0.0001,
    add_image_level_feature=True,
    aspp_with_separable_conv=True,
    decoder_output_stride=4,
    decoder_use_separable_conv=True,
    atrous_rates=None,
    fine_tune_batch_norm=False,
    image_pyramid=None,
    depth_multiplier=1.0):
  # TODO(rui): Add multiple resolutions.
  if image_pyramid == []:
    image_pyramid = [1.0]
  features, end_points = feature_extractor.extract_features(
      images,
      output_stride=output_stride,
      multi_grid=None,
      depth_multiplier=depth_multiplier,
      model_variant=model_variant,
      weight_decay=weight_decay,
      reuse=None,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm,
      regularize_depthwise=False)

  batch_norm_params = {
    'is_training': is_training and fine_tune_batch_norm,
    'decay': 0.9997,
    'epsilon': 1e-5,
    'scale': True,
  }

  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      padding='SAME',
      stride=1,
      reuse=None):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      depth = 256
      branch_logits = []

      if add_image_level_feature:
        # If crop_size is None, we simply do global pooling.
        image_feature = tf.reduce_mean(features, axis=[1, 2])[:, tf.newaxis,
                        tf.newaxis]
        resize_height = tf.shape(features)[1]
        resize_width = tf.shape(features)[2]

        image_feature = slim.conv2d(
          image_feature, depth, 1, scope=IMAGE_POOLING_SCOPE)
        image_feature = tf.image.resize_bilinear(
          image_feature, [resize_height, resize_width], align_corners=True)
        branch_logits.append(image_feature)

      # Employ a 1x1 convolution.
      branch_logits.append(slim.conv2d(features, depth, 1,
                                       scope=ASPP_SCOPE + str(0)))

      if atrous_rates:
        # Employ 3x3 convolutions with different atrous rates.
        for i, rate in enumerate(atrous_rates, 1):
          scope = ASPP_SCOPE + str(i)
          if aspp_with_separable_conv:
            aspp_features = split_separable_conv2d(
              features,
              filters=depth,
              rate=rate,
              weight_decay=weight_decay,
              scope=scope)
          else:
            aspp_features = slim.conv2d(
              features, depth, 3, rate=rate, scope=scope)
          branch_logits.append(aspp_features)

      # Merge branch logits.
      concat_logits = tf.concat(branch_logits, 3)
      concat_logits = slim.conv2d(
        concat_logits, depth, 1, scope=CONCAT_PROJECTION_SCOPE)
      features = slim.dropout(
        concat_logits,
        keep_prob=0.9,
        is_training=is_training,
        scope=CONCAT_PROJECTION_SCOPE + '_dropout')

  height = tf.shape(images)[1]
  width = tf.shape(images)[2]
  decoder_height = scale_dimension(height,
                                   1.0 / decoder_output_stride)
  decoder_width = scale_dimension(width,
                                  1.0 / decoder_output_stride)

  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      padding='SAME',
      stride=1,
      reuse=None):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with tf.variable_scope(DECODER_SCOPE, DECODER_SCOPE, [features]):
        feature_list = feature_extractor.networks_to_feature_maps[
            model_variant][feature_extractor.DECODER_END_POINTS]
        if feature_list is None:
          tf.logging.info('Not found any decoder end points.')
          return features
        else:
          decoder_features = features
          for i, name in enumerate(feature_list):
            decoder_features_list = [decoder_features]

            # MobileNet variants use different naming convention.
            if 'mobilenet' in model_variant:
              feature_name = name
            else:
              feature_name = '{}/{}'.format(
                  feature_extractor.name_scope[model_variant], name)
            decoder_features_list.append(
                slim.conv2d(
                    end_points[feature_name],
                    48,
                    1,
                    scope='feature_projection' + str(i)))
            # Resize to decoder_height/decoder_width.
            for j, feature in enumerate(decoder_features_list):
              decoder_features_list[j] = tf.image.resize_bilinear(
                  feature, [decoder_height, decoder_width], align_corners=True)
            decoder_depth = 256

            if decoder_use_separable_conv:
              decoder_features = split_separable_conv2d(
                  tf.concat(decoder_features_list, 3),
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv0')
              decoder_features = split_separable_conv2d(
                  decoder_features,
                  filters=decoder_depth,
                  rate=1,
                  weight_decay=weight_decay,
                  scope='decoder_conv1')
            else:
              num_convs = 2
              decoder_features = slim.repeat(
                  tf.concat(decoder_features_list, 3),
                  num_convs,
                  slim.conv2d,
                  decoder_depth,
                  3,
                  scope='decoder_conv' + str(i))

  kernel_size = 1
  scope_suffix = 'logits'
  if atrous_rates is None:
    atrous_rates = [1]
    if kernel_size != 1:
      raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                       'using aspp_with_batch_norm. Gets %d.' % kernel_size)
  else:
    atrous_rates.append(1)

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      reuse=None):
    with tf.variable_scope(LOGITS_SCOPE_NAME, LOGITS_SCOPE_NAME, [decoder_features]):
      branch_logits = []
      for i, rate in enumerate(atrous_rates):
        scope = scope_suffix
        if i:
          scope += '_%d' % i

        branch_logits.append(
            slim.conv2d(
                decoder_features,
                num_classes,
                kernel_size=kernel_size,
                rate=(rate, rate),
                activation_fn=None,
                normalizer_fn=None,
                scope=scope))

      return tf.add_n(branch_logits, name='merged_logits')


def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):
  """Splits a separable conv2d into depthwise and pointwise conv2d.

  This operation differs from `tf.layers.separable_conv2d` as this operation
  applies activation function between depthwise and pointwise conv2d.

  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.

  Returns:
    Computed features after split separable conv2d.
  """
  outputs = slim.separable_conv2d(
      inputs,
      None,
      kernel_size=kernel_size,
      depth_multiplier=1,
      rate=(rate, rate),
      weights_initializer=tf.truncated_normal_initializer(
          stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')