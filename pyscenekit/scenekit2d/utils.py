import torch
import PIL.Image
import numpy as np
from typing import List, Union

ImageInput = Union[
    str,
    PIL.Image.Image,
    np.ndarray,
    torch.FloatTensor,
    List[str],
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.FloatTensor],
]

TextInput = Union[str, List[str]]
