from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Mapping, Union

@dataclass
class DatasetConfig():
  name: Optional[str] = None
  data_dir: Optional[str] = None
  split: str = 'train'
  num_classes: List[] = 0
  num_examples: int = 0
  skip_label: Optional[int] = 0