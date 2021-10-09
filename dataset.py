""" Dataset module
"""
import cv2
import numpy as np
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Grayscale


class MaskDataset(Dataset):
    """ Masked faces dataset
        0 = 'no mask'
        1 = 'mask'
    """
    def __init__(self, dataFrame, transformations):
        self.dataFrame = dataFrame
        
        # self.transformations = Compose([
        #     ToPILImage(),
        #     Resize((100, 100)),
        #     Grayscale(num_output_channels=3),

        #     ToTensor(), # [0, 1]
        # ])

        self.transformations = transformations
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')
        
        row = self.dataFrame.iloc[key]
        image = cv2.imdecode(np.fromfile(row['image'], dtype=np.uint8),
                             cv2.IMREAD_UNCHANGED)
        return {
            'image': self.transformations(image),
            'mask': tensor(row['mask'], dtype=long), # pylint: disable=not-callable
        }
    
    def __len__(self):
        return len(self.dataFrame.index)
