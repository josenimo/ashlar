"""Illumination profiles as BaSiCPy writes them: float32, deflate, floating-point predictor."""

import numpy as np
import tifffile

from ashlar import reg


def test_load_basicpy_profile(tmp_path):
    # Decoding the floating-point predictor needs imagecodecs; without it the
    # whole histo pipeline failed at ASHLAR --dfp/--ffp.
    rng = np.random.default_rng(0)
    profile = rng.random((4, 64, 80), dtype=np.float32)
    path = tmp_path / "cycle1-ffp.ome.tif"
    tifffile.imwrite(path, profile, compression="zlib", predictor=True, ome=True)
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].predictor == tifffile.PREDICTOR.FLOATINGPOINT

    mosaic = reg.Mosaic.__new__(reg.Mosaic)
    mosaic.barrel_correction = None
    loaded = mosaic._load_single_profile(str(path), 4, (64, 80), "flat")

    np.testing.assert_array_equal(loaded, profile)
