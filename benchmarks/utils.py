import numpy as np


def upsample_array(input_array: np.ndarray, num_points: int) -> np.ndarray:
    """
    Upsample a 1D array to a specified number of points using linear interpolation.

    Args:
        input_array (np.ndarray) : The input array to be upsampled.
        num_points (int): The desired number of points in the upsampled array.

    Returns:
        upsampled_array (np.ndarray): 1D unsampled array with the specified number of points or the original
            array if num_points is less than or equal to the length of the input array.
    """
    input_array_length = len(input_array)
    if num_points <= input_array_length:
        return input_array

    original_points = np.arange(input_array_length)
    upsampled_points = np.linspace(0, input_array_length - 1, num_points)
    upsampled_array = np.interp(upsampled_points, original_points, input_array)
    return upsampled_array
