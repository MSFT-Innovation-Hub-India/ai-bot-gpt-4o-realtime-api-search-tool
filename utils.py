import numpy as np
import base64

def float_to_16bit_pcm(float32_array):
    """
    Converts a numpy array of float32 amplitude data to a numpy array in int16 format.
    :param float32_array: numpy array of float32
    :return: numpy array of int16
    """
    int16_array = np.clip(float32_array, -1, 1) * 32767
    return int16_array.astype(np.int16)


def base64_to_array_buffer(base64_string):
    """
    Converts a base64 string to a numpy array buffer.
    :param base64_string: base64 encoded string
    :return: numpy array of uint8
    """
    binary_data = base64.b64decode(base64_string)
    return np.frombuffer(binary_data, dtype=np.uint8)


def array_buffer_to_base64(array_buffer):
    """
    Converts a numpy array buffer to a base64 string.
    :param array_buffer: numpy array
    :return: base64 encoded string
    """
    if array_buffer.dtype == np.float32:
        array_buffer = float_to_16bit_pcm(array_buffer)
    elif array_buffer.dtype == np.int16:
        array_buffer = array_buffer.tobytes()
    else:
        array_buffer = array_buffer.tobytes()

    return base64.b64encode(array_buffer).decode("utf-8")


def merge_int16_arrays(left, right):
    """
    Merge two numpy arrays of int16.
    :param left: numpy array of int16
    :param right: numpy array of int16
    :return: merged numpy array of int16
    """
    if (
        isinstance(left, np.ndarray)
        and left.dtype == np.int16
        and isinstance(right, np.ndarray)
        and right.dtype == np.int16
    ):
        return np.concatenate((left, right))
    else:
        raise ValueError("Both items must be numpy arrays of int16")
