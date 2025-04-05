"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

from pathlib import Path

import numpy as np
import pandas as pd

import lsdb
from benchmarks.utils import upsample_array
from lsdb.core.search.box_search import box_filter
from lsdb.core.search.polygon_search import get_cartesian_polygon
from lsdb.types import CatalogTypeVar

TEST_DIR = Path(__file__).parent.parent / "tests"
DATA_DIR_NAME = "data"
SMALL_SKY_DIR_NAME = "small_sky"
SMALL_SKY_ORDER1 = "small_sky_order1"
SMALL_SKY_XMATCH_NAME = "small_sky_xmatch"
BENCH_DATA_DIR = Path(__file__).parent / "data"


def load_small_sky() -> CatalogTypeVar | None:
    """Load small sky catalog in the path `TEST_DIR`/`DATA_DIR_NAME`/`SMALL_SKY_DIR_NAME`.

    Returns:
        Dataset loaded by the `lsdb.read_hats` function.
    """
    return lsdb.read_hats(TEST_DIR / DATA_DIR_NAME / SMALL_SKY_DIR_NAME)


def load_small_sky_order1() -> CatalogTypeVar | None:
    """Load small sky order1 catalog in the path `TEST_DIR`/`DATA_DIR_NAME`/`SMALL_SKY_ORDER1`.

    Returns:
        Dataset loaded by the `lsdb.read_hats` function.
    """
    return lsdb.read_hats(TEST_DIR / DATA_DIR_NAME / SMALL_SKY_ORDER1)


def load_small_sky_xmatch():
    """Load small sky xmatch catalog in the path `TEST_DIR`/`DATA_DIR_NAME`/`SMALL_SKY_XMATCH_NAME`.

    Returns:
        Dataset loaded by the `lsdb.read_hats` function.
    """
    return lsdb.read_hats(TEST_DIR / DATA_DIR_NAME / SMALL_SKY_XMATCH_NAME)


def time_kdtree_crossmatch():
    """Time computations are prefixed with 'time'."""
    small_sky = load_small_sky()
    small_sky_xmatch = load_small_sky_xmatch()
    small_sky.crossmatch(small_sky_xmatch, require_right_margin=False).compute()


def time_polygon_search():
    """Time polygonal search using sphgeom"""
    small_sky_order1 = load_small_sky_order1().compute()
    # Upsample test catalog to 10,000 points
    catalog_ra = upsample_array(small_sky_order1["ra"].to_numpy(), 10_000)
    catalog_dec = upsample_array(small_sky_order1["dec"].to_numpy(), 10_000)
    # Define sky polygon to use in search
    vertices = [(300, -50), (300, -55), (272, -55), (272, -50)]
    polygon = get_cartesian_polygon(vertices)
    # Apply vectorized filtering on the catalog points
    polygon.contains(np.radians(catalog_ra), np.radians(catalog_dec))


def time_box_filter_on_partition():
    """Time box search on a single partition"""
    metadata = load_small_sky_order1().hc_structure
    mock_partition_df = pd.DataFrame(
        {
            metadata.catalog_info.ra_column: np.linspace(-1000, 1000, 100_000),
            metadata.catalog_info.dec_column: np.linspace(-90, 90, 100_000),
        }
    )
    box_filter(mock_partition_df, ra=(-20, 40), dec=(-90, 90), metadata=metadata.catalog_info)


def time_create_midsize_catalog():
    """Load midsize catalog in the path `BENCH_DATA_DIR/midsize_catalog/`.
    The midsize dataset contains 30_000 partitions at order 6.

    Returns:
        Dataset loaded by the `lsdb.read_hats` function.
    """
    return lsdb.read_hats(BENCH_DATA_DIR / "midsize_catalog")


def time_create_large_catalog():
    """Load midsize catalog in the path `BENCH_DATA_DIR/large_catalog/`.
    The large dataset contains 196_607 partitions at order 7.

    Returns:
        Dataset loaded by the `lsdb.read_hats` function.
    """
    return lsdb.read_hats(BENCH_DATA_DIR / "large_catalog")
