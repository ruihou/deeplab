"""Prepares the data used for DeepLab training/evaluation."""
import tensorflow as tf
from core import feature_extractor
from utils import preprocess_utils


# The probability of flipping the images and labels
# left-right during training
_PROB_OF_FLIP = 0.5


def preprocess_image_and_label(image,
                               label,
                               ignore_label=255,
                               is_training=True,
                               model_variant=None,
                               crop_height=None,
                               crop_width=None,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.0,
                               max_scale_factor=1.0,
                               scale_factor_step_size=0.0,
                               force_valid_scale=False):
  """Preprocesses the image and label.

  Args:
    image: Input image.
    label: Ground truth annotation label.
    options: A PreprocessorOptions message.
    ignore_label: The label value which will be ignored for training and
      evaluation.
    is_training: If the preprocessing is used for training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images.

  Returns:
    original_image: Original image (could be resized).
    processed_image: Preprocessed image.
    label: Preprocessed ground truth segmentation label.

  Raises:
    ValueError: Ground truth label not provided during training.
  """
  if is_training and label is None:
    raise ValueError('During training, label must be provided.')
  if model_variant is None:
    tf.logging.warning('Default mean-subtraction is performed. Please specify '
                       'a model_variant. See feature_extractor.network_map for '
                       'supported model variants.')

  # Keep reference to original image.
  if crop_height is None:
    crop_height = tf.shape(image)[1]
  if crop_width is None:
    crop_width = tf.shape(image)[2]

  processed_image = tf.cast(image, tf.float32)
  if label is not None:
    label = tf.cast(label, tf.int32)

  # Resize image and label to the desired range.
  if min_resize_value is not None and \
      max_resize_value is not None and \
      resize_factor is not None:
    [processed_image, label] = (
        preprocess_utils.resize_to_range(
            image=processed_image,
            label=label,
            min_size=min_resize_value,
            max_size=max_resize_value,
            factor=resize_factor,
            align_corners=True))

  # Data augmentation by randomly scaling the inputs.
  if is_training:
    scale = preprocess_utils.get_random_scale(
        min_scale_factor,
        max_scale_factor,
        scale_factor_step_size)
    processed_image, label = preprocess_utils.randomly_scale_image_and_label(
        processed_image, label, scale)

  # Pad image and label to have dimensions >= [crop_height, crop_width]
  image_shape = tf.shape(processed_image)
  image_height = image_shape[0]
  image_width = image_shape[1]

  target_height = image_height + tf.maximum(
      crop_height - image_height, 0)
  target_width = image_width + tf.maximum(crop_width - image_width, 0)

  # Pad image with mean pixel value.
  mean_pixel = tf.reshape(
      feature_extractor.mean_pixel(model_variant), [1, 1, 3])
  processed_image = preprocess_utils.pad_to_bounding_box(
      processed_image, 0, 0, target_height, target_width, mean_pixel)

  if label is not None:
    label = preprocess_utils.pad_to_bounding_box(
        label, 0, 0, target_height, target_width, ignore_label)

  # Randomly crop the image and label.
  if is_training and label is not None:
    processed_image, label = preprocess_utils.random_crop(
        [processed_image, label], crop_height, crop_width)
  else:
    processed_image, label = preprocess_utils.central_crop(
      [processed_image, label], crop_height, crop_width)

  if is_training:
    # Randomly left-right flip the image and label.
    processed_image, label, _ = preprocess_utils.flip_dim(
        [processed_image, label], _PROB_OF_FLIP)

  return processed_image, label
