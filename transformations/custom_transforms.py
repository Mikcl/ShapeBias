import difference_of_gaussian_transformation
import continuous_wavelet_transformation
import numpy as np

class DOG(object):
    '''
        Applies a Difference of Gaussian transformation on a image.
    '''
    def __init__(self, sigma=1.0, k=2.0):
        self.sigma = sigma
        self.k = k

    def __call__(self, image):
        '''
            ### Arguments:
                image: 'PIL' - multichannel image. 
            ### Return:
                list(list(float)) - difference of gaussian with predefined params applied.
        '''
        return difference_of_gaussian_transformation.dog(np.array(image), self.sigma, self.k)


class Gabor(object):
    '''
        Applies 2D continuous wavelet transformation at defined scales and orientations. 
    '''

    def __init__(self, scales=[2.0], orientations=[3]):
        self.scales = scales
        self.orientations = orientations

    def __call__(self, image):
        '''
            Apply transformation at 8 orientations and two scales. 

            ### Arguments:
                image: 'PIL' - multichannel image. 
            ### Return:
                list(list(float)) - gabor wavelet transformed image with X channels.
                            where X = 3 *|orientations| * |scales|
        '''
        image = np.array(image)
        transformed = []
        for channel in range(image.shape[2]):

            cwt = continuous_wavelet_transformation.cwt_2d(image[:,:,channel], 'gabor', multiplier=1., scales=self.scales, orientations=self.orientations)
            # take cwt.imag or cwt.real ?
            cwt = cwt.imag

            for i in range(cwt.shape[2]):
                transformed.append(np.uint8(cwt[:,:,i]))

        transformed = np.stack(transformed, axis=2)
        return transformed