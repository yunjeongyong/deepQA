import numpy as np
import scipy.misc as m


def error_map(img1, img2, epsilon=1.):
    assert img1.shape == img2.shape, 'Two inputs should have the same shape!'
    assert len(img1.shape) == 2, 'Inputs should be the grayscale.'

    # Higher value means lower distance
    # range: [0, 1]
    return np.log(1/(((img1-img2)**2)+epsilon/(255**2)))/np.log((255**2)/epsilon)


def low_frequency_sub(img, scale=4):
    assert len(img.shape) == 2, 'Inputs should be the grayscale.'

    # print((int(img.shape[0] / scale), int(img.shape[1] / scale)))
    # print('low', img.shape)
    img_resize = m.imresize(img, (int(img.shape[0] / scale), int(img.shape[1] / scale)),
                            interp='nearest')
    img_resize = m.imresize(img_resize, (img.shape[0], img.shape[1]),
                            interp='nearest')

    return (img-img_resize)/255.
