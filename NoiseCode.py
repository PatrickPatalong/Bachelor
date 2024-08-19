import numpy as np
import random


# WhiteNoises

def whitenoise(im_flat, im, mode, sigma, eta, noisemin=0, noisemax=1):
    if mode == "gaussian":
        whitenoise = np.zeros(im_flat.shape[0])
        for k in range(im_flat.shape[0]):
            whitenoise[k] = eta * random.gauss(0, sigma ** 2)
            im_flat[k] = im_flat[k] + whitenoise[k]
        return im_flat, whitenoise

    if mode == "specklenoise":
        gauss = np.random.normal(noisemin, noisemax, im_flat.size[0])  # Whitenoise
        result = im + im * gauss  # Zusammenmixen der beiden Images
        return result, gauss

    else:
        print("gaussian oder specklenoise für WHITENOISEMODE benutzen.")


# Blur

def blur(im, Convolution):
    u_del = Convolution.dot(im)
    return u_del