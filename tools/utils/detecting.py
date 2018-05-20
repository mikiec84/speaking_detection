import dlib


def dlib_c_rect(r):
    return dlib.rectangle(*map(int, [r.left(), r.top(), r.right(), r.bottom()]))


import numpy
clip = numpy.clip
