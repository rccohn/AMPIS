import numpy as np
import torch
from typing import List, Union


class RLEMasks:
    """
    Class for adding RLE masks to Instances object
    """
    def __init__(self, masks):
        """

        Args:
            masks: n element list(dic) of RLE encoded masks
        """
        self.masks = masks

    def __getitem__(self, item: Union[int, slice, List[int], List[bool],
                                      torch.BoolTensor, np.ndarray]):
        if type(item) == int:
            return RLEMasks(self.masks[item])

        elif type(item) == torch.BoolTensor:
            return RLEMasks([mask for mask, bool_ in zip(self.masks, item) if bool_])

        elif type(item) == np.ndarray:
            if item.dtype == np.bool:
                return RLEMasks([mask for mask, bool_ in zip(self.masks, item) if bool_])

        elif type(item) == slice:
            return RLEMasks(self.masks[item])

        elif type(item) == list:
            if type(item[0]) == bool:
                return RLEMasks([mask for mask, bool_ in zip(self.masks, item) if bool_])

        else:
            # list, (tuple, array, tensor, etc) of integer indices
            return RLEMasks([self.masks[idx] for idx in item])

    def __len__(self):
        return(len(self.masks))



