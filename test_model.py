import numpy as np
from model import one_hot_encode_shape, classify_arr

def test_one_hot_encode_shape():
    arr = np.array([[3,7,2],[5,1,4]])
    enc_arr = one_hot_encode_shape(arr, '', 3)
    print('enc_arr\n', enc_arr)
    assert enc_arr.shape == (2,3,3)

def test_classify_arr():
    arr = np.array([3,9,13,2,8,4,11])
    cls = classify_arr(arr, np.array([0.2, 0.8, 1.0]))
    print(cls)

if __name__ == '__main__':
    test_classify_arr()