import numpy as np

from model import rescale_quantized_weight


def test_rescale():
    weight = np.arange(42).reshape((6, 7)).astype(np.float16)

    # Each row of scales is applied to
    # three consecutive rows of weight.
    scales = np.arange(2 * 7).reshape((2, 7)).astype(np.int32)

    rescaled_array = rescale_quantized_weight(weight, scales)
    assert rescaled_array.shape == weight.shape
    assert rescaled_array[:, 0].flatten().tolist() == [
        0 * 0,
        0 * 7,
        0 * 14,
        7 * 21,
        7 * 28,
        7 * 35,
    ]
    assert rescaled_array.dtype == np.int32
