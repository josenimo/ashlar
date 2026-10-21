"""QC output tests on a small synthetic 3x3 tile, 3-cycle dataset."""

import csv
import json

import numpy as np
import pytest
import scipy.ndimage

from ashlar import qc, reg, report


TILE = 200
STEP = 170  # 15 % overlap
GRID = 3


class ArrayMetadata(reg.Metadata):
    def __init__(self, positions):
        self._pos = np.asarray(positions, float)

    @property
    def _num_images(self):
        return len(self._pos)

    @property
    def num_channels(self):
        return 2

    @property
    def pixel_size(self):
        return 0.5

    @property
    def pixel_dtype(self):
        return np.dtype(np.uint16)

    def tile_position(self, i):
        return self._pos[i]

    def tile_size(self, i):
        return np.array([TILE, TILE])


class ArrayReader(reg.Reader):
    """Cuts tiles from a large image; channel 0 is texture, 1 is noise."""

    def __init__(self, image, positions, true_positions, noise_seed):
        self.image = image
        self.metadata = ArrayMetadata(positions)
        self.true_positions = np.round(true_positions).astype(int)
        self.noise_seed = noise_seed

    def read(self, series, c):
        if c < 0:
            c += self.metadata.num_channels
        y, x = self.true_positions[series]
        if c == 0:
            return self.image[y:y + TILE, x:x + TILE].copy()
        rng = np.random.default_rng(self.noise_seed * 100 + series)
        return rng.integers(0, 1000, (TILE, TILE), dtype=np.uint16)


def texture(shape, seed=0):
    rng = np.random.default_rng(seed)
    img = scipy.ndimage.gaussian_filter(rng.random(shape), 2)
    img -= img.min()
    return (img / img.max() * 60000).astype(np.uint16)


def make_cycle(image, offset, angle, seed, stage_scale=1.0, jitter=3):
    true_grid = np.array(
        [[r * STEP, c * STEP] for r in range(GRID) for c in range(GRID)], float
    )
    # stage_scale < 1 means the stage reports wider spacing than the pixels show.
    nominal = true_grid / stage_scale
    rng = np.random.default_rng(seed)
    true = true_grid + 40 + rng.uniform(-jitter, jitter, nominal.shape) + offset
    if angle:
        image = scipy.ndimage.rotate(image, angle, reshape=False, order=1)
    return ArrayReader(image, nominal, true, seed)


@pytest.fixture(scope="module")
def aligners():
    image = texture((STEP * GRID + 200, STEP * GRID + 200))
    readers = [
        make_cycle(image, (0, 0), 0, 1),
        make_cycle(image, (12, -7), 0, 2),
        make_cycle(image, (5, 5), 0.8, 3),
    ]
    ea = reg.EdgeAligner(readers[0], channel=0, max_shift=20)
    ea.run()
    las = []
    for r in readers[1:]:
        la = reg.LayerAligner(r, ea, channel=0, max_shift=20)
        la.run()
        las.append(la)
    return [ea, *las]


@pytest.fixture(scope="module")
def qc_dir(aligners, tmp_path_factory):
    path = tmp_path_factory.mktemp("qc")
    qc.write_qc(
        path, aligners, filepaths=["a", "b", "c"], args={"x": 1},
        warnings=["something odd"],
    )
    return path


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_files_written(qc_dir):
    names = {p.name for p in qc_dir.iterdir()}
    assert {
        "qc.json", "edges.csv", "tiles.csv", "null_errors.csv",
        "thumb_cycle0.jpg", "thumb_cycle1.jpg", "thumb_cycle2.jpg",
        "overlay_cycle1.jpg", "overlay_cycle2.jpg",
    } <= names


def test_summary_counts(qc_dir):
    summary = json.loads((qc_dir / "qc.json").read_text())
    st = summary["stitching"]
    assert summary["schema_version"] == qc.SCHEMA_VERSION
    assert summary["warnings"] == ["something odd"]
    assert st["n_tiles"] == GRID * GRID
    # 3x3 grid: 6 horizontal + 6 vertical neighbour pairs.
    assert st["n_edges"] == 12
    assert st["n_edges_accepted"] == 12
    assert st["n_components"] == 1
    assert st["largest_component_fraction"] == 1
    assert st["stage_model"]["scale_x"] == pytest.approx(1, abs=0.05)
    assert len(summary["registration"]) == 2
    assert set(summary["images"]["overlays"]) == {
        "overlay_cycle1.jpg", "overlay_cycle2.jpg"
    }


def test_registration_recovers_truth(qc_dir):
    summary = json.loads((qc_dir / "qc.json").read_text())
    c1, c2 = summary["registration"]
    # Cycle 1 tiles see content moved by (+12, -7) px. The coarse offset is
    # only accurate up to the per-cycle stage jitter (+/-3 px).
    np.testing.assert_allclose(c1["cycle_offset_px"], [12, -7], atol=6)
    # Thumbnail rotation detection is noisy on this small synthetic texture.
    assert abs(c1["cycle_angle"]) < 0.3
    assert c1["n_discard"] == c1["n_discard_background"] + c1["n_discard_shift"]
    assert c1["n_kept_like_null"] == 0
    assert c2["angle_refined"] is True
    assert abs(c2["cycle_angle"]) == pytest.approx(0.8, abs=0.2)


def test_final_positions_match_truth(aligners):
    ea, la1, _ = aligners
    # Relative placement of the same tile in two cycles must equal the
    # difference of where its content truly came from.
    # Discarded tiles are model-placed and keep their stage jitter.
    truth = la1.reader.true_positions - ea.reader.reader.true_positions
    kept = ~la1.discard
    assert kept.sum() >= 3
    np.testing.assert_allclose(
        (la1.positions - ea.positions)[kept], truth[kept], atol=1
    )


def test_tiles_and_edges_tables(qc_dir):
    tiles = read_csv(qc_dir / "tiles.csv")
    assert list(tiles[0]) == qc.TILE_COLUMNS
    assert len(tiles) == 3 * GRID * GRID
    cycle0 = [t for t in tiles if t["cycle"] == "0"]
    assert all(t["error"] == "" and t["component"] == "0" for t in cycle0)
    edges = read_csv(qc_dir / "edges.csv")
    assert len(edges) == 12
    assert sum(int(e["in_tree"]) for e in edges) == GRID * GRID - 1
    assert {e["reject_reason"] for e in edges} == {"none"}


def test_failed_cycle_is_flagged(aligners, tmp_path):
    # Aligning on the pure-noise channel must fail registration for most tiles.
    ea = aligners[0]
    image = texture((STEP * GRID + 200, STEP * GRID + 200))
    bad = make_cycle(image, (0, 0), 0, 9)
    la = reg.LayerAligner(bad, ea, channel=1, max_shift=20)
    la.run()
    summary = qc.write_qc(tmp_path, [ea, la])
    reg_qc = summary["registration"][0]
    # Each usable (foreground) tile is either dropped for landing far from
    # the model or kept with an error no better than random strips.
    usable = reg_qc["n_tiles"] - reg_qc["n_discard_background"]
    assert usable > 0
    assert reg_qc["n_discard_shift"] + reg_qc["n_kept_like_null"] == usable


def test_rejected_edges_recorded(tmp_path):
    image = texture((STEP * GRID + 200, STEP * GRID + 200))
    noisy = reg.EdgeAligner(make_cycle(image, (0, 0), 0, 1), channel=1)
    noisy.run()
    summary = qc.write_qc(tmp_path, [noisy])
    st = summary["stitching"]
    assert st["n_edges_accepted"] < st["n_edges"]
    assert st["n_components"] > 1
    assert st["edges_like_null_fraction"] > 0.3
    json.loads((tmp_path / "qc.json").read_text())


def test_json_is_strict():
    # NaN angles (background tiles) and inf errors must not leak into JSON.
    summary = qc.clean({"a": float("nan"), "b": np.inf, "c": np.float32(1)})
    assert summary == {"a": None, "b": None, "c": 1.0}


def test_pdf_report(aligners, tmp_path):
    path = tmp_path / "report.pdf"
    report.generate_report(str(path), aligners)
    assert path.read_bytes()[:4] == b"%PDF"


def test_model_prediction_is_unbiased_under_stage_scale():
    # The stitching model is fitted on top-left positions; applying it to tile
    # centres used to add (scale - 1) * size / 2 to every prediction.
    image = texture((STEP * GRID + 200, STEP * GRID + 200))
    ea = reg.EdgeAligner(make_cycle(image, (0, 0), 0, 1, 0.95, 0), channel=0)
    ea.run()
    assert ea.lr.coef_[0, 0] < 0.98
    la = reg.LayerAligner(
        make_cycle(image, (12, -7), 0, 2, 0.95, 0), ea, channel=0, max_shift=20
    )
    la.run()
    usable = ~(la.discard_error | la.discard_background)
    assert usable.sum() >= 3
    # Without stage jitter every tile sits on the model; the bias was ~5 px.
    assert la.model_distance[usable].max() < 1


def test_pdf_report_survives_non_latin1_warnings(aligners, tmp_path):
    # Ashlar's own unit warning uses Greek mu, which the core PDF fonts lack.
    path = tmp_path / "report.pdf"
    summary = qc.summarize(
        aligners, warnings=["Stage coordinates' unit is undefined; assuming μm."]
    )
    report.generate_report(str(path), aligners, summary)
    assert path.read_bytes()[:4] == b"%PDF"
