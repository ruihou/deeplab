import collections

DatasetOption = collections.namedtuple('DatasetOption',
                                       ['name',
                                        'num_classes',
                                        'ignore_label',
                                        'label_bias',
                                        'label_scale',
                                        'image_format',
                                        'label_format',
                                        'num_samples'],
                                       verbose=False)

_datasets_map = {
    'adechallenge': DatasetOption(name='adechallenge',
                                  num_classes=150,
                                  ignore_label=-1,
                                  label_bias=0,
                                  label_scale=1,
                                  image_format='jpg',
                                  label_format='png',
                                  num_samples={'training': 20210,
                                               'validation': 2000,
                                               'test': 500,
                                               'trainval': 22210}),
    'davis16': DatasetOption(name='davis16',
                             num_classes=2,
                             ignore_label=-1,
                             label_bias=0,
                             label_scale=255,
                             image_format='jpg',
                             label_format='png',
                             num_samples={'train': 2079,
                                          'val': 1376,
                                          'trainval': 2079 + 1376}),
    'jhmdb': DatasetOption(
        name='jhmdb',
        num_classes=2,
        ignore_label=-1,
        label_bias=0,
        label_scale=1,
        image_format='png',
        label_format='png',
        num_samples={'split1_train': 22712,
                     'split1_val': 9126,
                     'split1': 22712 + 9126,
                     'split2_train': 22881,
                     'split2_val': 8957,
                     'split3_train': 22793,
                     'split3_val': 9045}),
}


def ignore_label(name):
  return _datasets_map[name].ignore_label


def num_classes(name):
  return _datasets_map[name].num_classes


def label_bias(name):
  return _datasets_map[name].label_bias


def label_scale(name):
  return _datasets_map[name].label_scale


def image_format(name):
  return _datasets_map[name].image_format


def label_format(name):
  return _datasets_map[name].label_format

def num_samples(name, split):
  return _datasets_map[name].num_samples[split]
