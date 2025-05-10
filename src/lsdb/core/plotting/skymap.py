from __future__ import annotations

from collections.abc import Callable
from typing import Any

import hats.pixel_math.healpix_shim as hp
import nested_pandas as npd
import numpy as np
from dask import delayed
from hats.pixel_math import HealpixPixel, spatial_index_to_healpix


@delayed
def perform_inner_skymap(
    partition: npd.NestedFrame,
    func: Callable[[npd.NestedFrame, HealpixPixel], Any],
    pixel: HealpixPixel,
    target_order: int,
    default_value: Any = 0,
    **kwargs,
) -> np.ndarray:
    """Splits a partition into smaller Healpix pixels at a specified target order and applies a given function
    to each of the new pixels.

    Args:
        partition (npd.NestedFrame): The input data partition, indexed by spatial indices.
        func (Callable[[npd.NestedFrame, HealpixPixel], Any]): A function to apply to each new Healpix pixel.
            The function takes a subset of the partition and a HealpixPixel object as arguments.
        pixel (HealpixPixel): The Healpix pixel representing the current partition.
        target_order (int): The target Healpix order to which the partition will be split.
        default_value (Any, optional): The default value to fill in the resulting array for pixels
            where no data is available. Defaults to 0.
        **kwargs: Additional keyword arguments to pass to the `func`.

    Returns:
        np.ndarray: A 1D array representing the results of applying the function to each new Healpix pixel
        at the target order. The array is filled with `default_value` for pixels with no data.

    Notes:
        - The function assumes that the input partition is indexed by spatial indices that can be converted to
        Healpix pixels.
        - The `spatial_index_to_healpix` function is used to map spatial indices to Healpix pixels.
        - The resulting array is aligned with the Healpix pixel indices at the target order.
    """
    delta_order = target_order - pixel.order
    img = np.full(1 << 2 * delta_order, fill_value=default_value)

    if len(partition) == 0:
        return img

    spatial_index = partition.index.to_numpy()
    order_pixels = spatial_index_to_healpix(spatial_index, target_order=target_order)

    def apply_func(df):
        # gets the healpix pixel of the partition using the spatial index
        p = spatial_index_to_healpix([df.index.to_numpy()[0]], target_order=target_order)[0]
        return func(df, HealpixPixel(target_order, p), **kwargs)

    gb = partition.groupby(order_pixels, sort=False).apply(apply_func)
    min_pixel_value = pixel.pixel << 2 * delta_order
    img[gb.index.to_numpy() - min_pixel_value] = gb.to_numpy(na_value=default_value)
    return img


def compute_skymap(
    pixel_map: dict[HealpixPixel, Any], order: int | None = None, default_value: Any = 0.0
) -> np.ndarray:
    """Returns a histogram map of healpix_pixels to values.

    Args:
        pixel_map(Dict[HealpixPixel, Any]): A dictionary of healpix pixels and their values
        order (int): The order to make the histogram at (default None, uses max order in pixel_map)
        default_value: The value to use at pixels that aren't covered by the pixel_map (default 0)
    """

    pixels = list(pixel_map.keys())
    if len(pixels) == 0:
        npix = hp.order2npix(order) if order is not None else hp.order2npix(0)
        return np.full(npix, default_value)
    hp_orders = np.vectorize(lambda x: x.order)(pixels)
    hp_pixels = np.vectorize(lambda x: x.pixel)(pixels)
    if order is None:
        order = np.max(hp_orders)
    npix = hp.order2npix(order)
    img = np.full(npix, default_value)
    dorders = order - hp_orders
    values = [pixel_map[x] for x in pixels]
    starts = hp_pixels << (2 * dorders)
    ends = (hp_pixels + 1) << (2 * dorders)

    def set_values(start, end, value):
        if value is not None:
            img[np.arange(start, end)] = value

    for s, e, v in zip(starts, ends, values, strict=False):
        set_values(s, e, v)

    return img
