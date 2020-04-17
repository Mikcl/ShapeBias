import numpy as np
import math

'''
    Generate various 2D wavelets.
'''

def gabor(x, y, u=1, v=2.0, k_max=np.pi, f=math.sqrt(2), d2=(2*np.pi)**2):
    '''
        2D Gabor Wavelet. 

        ### Arguments:
            x: list(list(float)) - reciprocal space in x plane
            y: list(list(float)) - reciprocal space in y plane
            u: int -  orientation from 1-8 inclusive indicating direction of wavelet. 
            v: float - scale of wavelet. 
        ### Returns:
            list(list( (i,j) )) - real and imaginary coefficients of two dimensional Gabor Wavelet. 

    '''
    k = (k_max / (f ** v)) * np.exp(1j * u * (np.pi)/8) # wave vector
    k2 = abs(k) ** 2

    return (k2/d2) * np.exp( (-0.5 * k2 * (x**2 * y**2)) / d2) * \
        (np.exp(k.real * x + k.imag * y) - np.exp(-0.5 *d2))

def mexican_hat(x, y, sigma_y=1, sigma_x=1, order=2):
    '''
    2D Mexican Hat Wavelet. 

        ### Arguments:
            x: list(list(float)): reciprocal space in x plane
            y: list(list(float)): reciprocal space in y plane
        ### Returns:
            list(list(float)) - 2D Mexican Hat Wavelet coefficients. 
    '''
    return -(2 * np.pi) * (x**2 + y**2)**(order / 2) * \
           np.exp(-((sigma_x * x)**2 + (sigma_y * y)**2) / 2)


def dog(x, y, alpha=1.25):
    '''
        Derivative of Gaussian Wavelet
    '''
    m = (x**2 + y**2) / 2
    return -np.exp(-m) + np.exp(-alpha**2 * m)


wavelets = dict(
    mexican_hat=mexican_hat,
    dog=dog,
    gabor=gabor
)
