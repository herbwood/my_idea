import torch
from torch import Tensor
from typing import List, Tuple


class ImageList:

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]):

        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> 'ImageList':
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)