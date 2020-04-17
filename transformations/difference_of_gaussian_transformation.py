from skimage import filters
from PIL import Image
import numpy as np

def dog(image, sigma=1.0, k=2.0):
    '''
        Difference of Gaussian transformation on image .

        Args:
            image: list(list(float)) - image to apply the transformation, multi channel. 
            sigma: float - value to apply to gaussian formula. 
            k: float - scalar to apply to sigma. 
        Returns:
            list(list(float)) - image with difference of gaussian transformation applied. 

    '''
    s1 = filters.gaussian(image,k*sigma, multichannel=True)
    s2 = filters.gaussian(image, sigma, multichannel=True)

    dog = s1 - s2
    return np.uint8(dog*255)
