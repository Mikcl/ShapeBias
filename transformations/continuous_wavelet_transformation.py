import numpy as np
from functools import lru_cache

from mother_wavelets import wavelets

'''
    A simple implementation of 2D continuous wavelet transform,
    using the fast fourier transform method. 

    Inspired by https://github.com/LeonArcher/py_cwt2d - MIT licence. 
'''

def get_wavelet_mask(wavelet, freq_x, freq_y, **kwargs):
    '''
        Get 2D wavelet defined with the parameters passed.

        ### Arguments:
            wavelet: str - name of wavelet to use
            freq_x: list(list(float)) - reciprocal space in x plane. 
            freq_y: list(list(float)) - reciprocal space in y plane. 
        ### Return:
            list(list(num)) - 2D wavelet
    '''
    return wavelets[wavelet](freq_x, freq_y, **kwargs)

@lru_cache(25)
def create_frequency_plane(shape):
    '''
        ### Arguments: 
            shape - (int, int): height and width of an image.
        ### Returns:
            (list(list(float)) , list(list(float))): reciprocal space in x and y plane. 
    '''
    h, w = shape
    w_2 = (w - 1) // 2
    h_2 = (h - 1) // 2
    # 2
    scaler = 2.0
    w_pulse = scaler * np.pi / w * np.hstack((np.arange(0, w_2 + 1), np.arange(w_2 - w + 1, 0)))
    h_pulse = scaler * np.pi / h * np.hstack((np.arange(0, h_2 + 1), np.arange(h_2 - h + 1, 0)))

    xx, yy = np.meshgrid(w_pulse, h_pulse, indexing='xy')
    return xx, yy 

def convolve_wavelet(image, wavelet, xx, yy, multiplier, **wavelet_args):
    '''
        Apply convolution using FFT procedure to get wavelet coefficients of image.

        ### Arguments:
            image: list(list(float)) - image which has had the 2D FFT (np.fft2) method applied to it 
            wavelet: str- name of wavelet to apply. 
            xx: list(list(float)) - reciprocal space in x plane.
            yy: list(list(float)) - reciprocal space in y plane. 
            multiplier: float: value to multiply frequency planes by
        ### Return:
            list(list(number)) - image convolved with two dimensional wavelet. 
    '''

    mask = multiplier * get_wavelet_mask(wavelet, multiplier * xx, multiplier * yy, **wavelet_args)
    return np.fft.ifft2( image * mask ) 
    

def cwt_2d(image, wavelet, multiplier=1.4, orientations=[1], scales=[2.0]):
    '''
        Apply 2D CWT on wavelet provided, at the orientations and scales provided.
        
        ### Arguments:
            image - list(list(float)): image to apply CWT to. 
            wavelet - str: name of wavelet to apply. 
            orientations - list(int): series of orientations to take (1-8) inclusive
            scales - list(float): scales of wavelet to apply. 
        ### Return:
            list(list(float)) - CWT coefficients of image applied at different scales and orientations.
                        leading to |orientations| * |scales| channels . NP ARRAY
    '''
    x_image = np.fft.fft2(image)
    xx, yy = create_frequency_plane(x_image.shape)
    
    cwt = []
    for u in orientations:
        for v in scales:
            transformed = convolve_wavelet(x_image, wavelet, xx, yy, multiplier, u=u, v=v)
            cwt.append(transformed)

    cwt = np.stack(cwt, axis=2)

    return cwt