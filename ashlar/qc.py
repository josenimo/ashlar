"""Machine-readable stitching and registration QC output.

write_qc() dumps the state of a finished EdgeAligner (cycle 0) and its
LayerAligners (cycles 1+) as plain JSON, CSV and JPEG files so that pipelines
can inspect alignment quality without importing ashlar. Coordinates are
[y, x] in pixels unless a name ends in _um; errors are -log(NCC), lower is
better. Non-finite values are written as null (JSON) or empty (CSV).
"""

import csv
import json
import math
import pathlib

import networkx as nx
import numpy as np
from PIL import Image
import skimage.transform

from . import __version__ as _version
from . import utils


SCHEMA_VERSION = 1
IMAGE_MAX_SIZE = 1000
JPEG_QUALITY = 85


def write_qc(path, aligners, filepaths=None, args=None, warnings=None):
    """Write QC files for one stitched+registered mosaic into directory path."""
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    edge_aligner, *layer_aligners = aligners
    summary = summarize(aligners, filepaths, args, warnings)
    write_edges(path / "edges.csv", edge_aligner)
    write_tiles(path / "tiles.csv", edge_aligner, layer_aligners)
    write_column(
        path / "null_errors.csv", "error", edge_aligner.errors_negative_sampled
    )
    summary["images"] = write_images(path, edge_aligner, layer_aligners)
    with open(path / "qc.json", "w") as f:
        json.dump(clean(summary), f, indent=2, allow_nan=False)
        f.write("\n")
    return summary


def summarize(aligners, filepaths=None, args=None, warnings=None):
    """Return the qc.json summary dict (numpy values not yet cleaned)."""
    edge_aligner, *layer_aligners = aligners
    return {
        "schema_version": SCHEMA_VERSION,
        "ashlar_version": _version,
        "args": args or {},
        "inputs": [str(p) for p in filepaths] if filepaths else [],
        "pixel_size_um": float(edge_aligner.metadata.pixel_size),
        "tile_size": [int(x) for x in edge_aligner.metadata.size],
        "mosaic_shape": [int(x) for x in edge_aligner.mosaic_shape],
        "warnings": list(warnings or []),
        "stitching": stitching_summary(edge_aligner),
        "registration": [
            registration_summary(la, cycle)
            for cycle, la in enumerate(layer_aligners, 1)
        ],
    }


def stitching_summary(ea):
    n_tiles = ea.metadata.num_images
    reasons = list(ea.edge_reject.values())
    raw_errors = ea.all_errors
    null = ea.errors_negative_sampled
    components = sorted(
        nx.connected_components(ea.spanning_tree), key=len, reverse=True
    )
    model_dist = model_distance_um(ea)
    finite_raw = raw_errors[np.isfinite(raw_errors)]
    return {
        "n_tiles": n_tiles,
        "n_edges": len(reasons),
        "n_edges_accepted": reasons.count("none"),
        "n_edges_rejected_error": reasons.count("error") + reasons.count("both"),
        "n_edges_rejected_shift": reasons.count("shift") + reasons.count("both"),
        "max_error": ea.max_error,
        "max_shift_um": ea.max_shift,
        "alpha": ea.alpha,
        "null_error_percentiles": percentiles(null, [1, 5, 50]),
        "edge_error_percentiles": percentiles(finite_raw, [5, 50, 95]),
        # Share of real tile pairs scoring no better than a typical random
        # (non-overlapping) pair, i.e. pairs that look like noise.
        "edges_like_null_fraction": (
            float(np.mean(finite_raw >= np.median(null)))
            if len(null) and len(finite_raw) else None
        ),
        "n_components": len(components),
        "largest_component_fraction": len(components[0]) / n_tiles,
        "n_tiles_outside_largest_component": n_tiles - len(components[0]),
        "n_isolated_tiles": sum(
            1 for _, d in ea.spanning_tree.degree if d == 0
        ) if n_tiles > 1 else 0,
        "stage_model": stage_model(ea.lr.coef_),
        "model_residual_um_percentiles": percentiles(model_dist, [50, 95, 100]),
        "n_foreground_tiles": int(np.sum(ea.foreground)),
        "foreground_threshold": getattr(ea, "foreground_threshold", None),
        "timings_s": getattr(ea, "timings", {}),
    }


def registration_summary(la, cycle):
    px = la.metadata.pixel_size
    n = la.metadata.num_images
    kept = ~la.discard
    usable = ~(la.discard_error | la.discard_background)
    angles = getattr(la, "angles", None)
    finite_angles = (
        angles[np.isfinite(angles)] if angles is not None else np.empty(0)
    )
    residual_um = la.model_distance * px
    return {
        "cycle": cycle,
        "n_tiles": n,
        "channel": la.channel,
        "cycle_offset_px": la.cycle_offset,
        "cycle_offset_um": np.asarray(la.cycle_offset) * px,
        "cycle_angle_coarse": getattr(la, "cycle_angle_coarse", la.cycle_angle),
        "cycle_angle": la.cycle_angle,
        "angle_refined": angles is not None,
        "tile_angle_median": (
            np.median(finite_angles) if len(finite_angles) else None
        ),
        "tile_angle_mad": (
            np.median(np.abs(finite_angles - np.median(finite_angles)))
            if len(finite_angles) else None
        ),
        "n_discard": int(np.sum(la.discard)),
        "n_discard_error": int(np.sum(la.discard_error)),
        "n_discard_background": int(np.sum(la.discard_background)),
        "n_discard_shift": int(np.sum(la.discard_shift & usable)),
        "discard_fraction": float(np.mean(la.discard)),
        "offset_px": np.broadcast_to(la.offset, 2),
        # Distance of each registered tile from the stage model prediction,
        # for tiles that were not rejected as background or failed.
        "residual_um_percentiles": percentiles(
            residual_um[usable], [50, 95, 100]
        ),
        "kept_residual_um_percentiles": percentiles(
            residual_um[kept], [50, 95, 100]
        ),
        "kept_error_median": (
            np.median(la.errors[kept]) if kept.any() else None
        ),
        # LayerAligner never tests errors against the stitching null
        # threshold, so kept tiles scoring worse than it may be spurious.
        "n_kept_like_null": int(
            np.sum(kept & (la.errors > la.reference_aligner.max_error))
        ),
        "n_duplicate_reference_tiles": n - len(np.unique(la.reference_idx)),
        "timings_s": getattr(la, "timings", {}),
    }


def write_edges(path, ea):
    px = ea.metadata.pixel_size
    raw = dict(zip(list(ea._cache.keys()), ea.all_errors))
    fg = ea.foreground
    rows = []
    for t1, t2 in ea.neighbors_graph.edges:
        key = tuple(sorted((t1, t2)))
        shift, error = ea._cache[key]
        rows.append([
            key[0], key[1], shift[0], shift[1],
            np.linalg.norm(shift) * px, raw[key], error,
            ea.edge_reject[key], int(ea.spanning_tree.has_edge(*key)),
            int(fg[key[0]]), int(fg[key[1]]),
        ])
    write_csv(path, [
        "t1", "t2", "dy", "dx", "shift_um", "error_raw", "error_final",
        "reject_reason", "in_tree", "fg_t1", "fg_t2",
    ], rows)


TILE_COLUMNS = [
    "cycle", "tile", "nominal_y", "nominal_x", "y", "x", "shift_y", "shift_x",
    "error", "model_distance_um", "angle", "reference_tile", "discard_error",
    "discard_background", "discard_shift", "foreground", "foreground_energy",
    "component", "tree_degree",
]


def write_tiles(path, ea, layer_aligners):
    rows = []
    components = sorted(
        nx.connected_components(ea.spanning_tree), key=len, reverse=True
    )
    component = np.empty(ea.metadata.num_images, int)
    for ci, c in enumerate(components):
        component[list(c)] = ci
    model_dist = model_distance_um(ea)
    energy = getattr(ea, "foreground_energy", [None] * ea.metadata.num_images)
    for t in range(ea.metadata.num_images):
        rows.append([
            0, t, *ea.metadata.positions[t], *ea.positions[t], *ea.shifts[t],
            None, model_dist[t], None, None, None, None, None,
            int(ea.foreground[t]), energy[t], component[t],
            ea.spanning_tree.degree[t],
        ])
    for cycle, la in enumerate(layer_aligners, 1):
        px = la.metadata.pixel_size
        angles = getattr(la, "angles", None)
        ref_fg = la.reference_aligner.foreground[la.reference_idx]
        for t in range(la.metadata.num_images):
            rows.append([
                cycle, t, *la.metadata.positions[t], *la.positions[t],
                *la.shifts[t], la.errors[t], la.model_distance[t] * px,
                angles[t] if angles is not None else None,
                la.reference_idx[t], int(la.discard_error[t]),
                int(la.discard_background[t]), int(la.discard_shift[t]),
                int(ref_fg[t]), None, None, None,
            ])
    write_csv(path, TILE_COLUMNS, rows)


def write_images(path, ea, layer_aligners):
    """Write registered thumbnails and cycle overlays (JPEG); return names."""
    if not hasattr(ea.reader, "thumbnail"):
        return {}
    scale = ea.reader.thumbnail_scale
    shape = np.ceil(np.array(ea.mosaic_shape) * scale).astype(int)
    ref = registered_thumbnail(ea, shape, scale)
    images = {"thumbnails": [], "overlays": []}
    name = "thumb_cycle0.jpg"
    save_image(path / name, to_uint8(ref))
    images["thumbnails"].append(name)
    for cycle, la in enumerate(layer_aligners, 1):
        if not hasattr(la.reader, "thumbnail"):
            continue
        img = registered_thumbnail(la, shape, scale)
        name = f"thumb_cycle{cycle}.jpg"
        save_image(path / name, to_uint8(img))
        images["thumbnails"].append(name)
        name = f"overlay_cycle{cycle}.jpg"
        save_image(path / name, overlay(ref, img))
        images["overlays"].append(name)
    return images


def registered_thumbnail(aligner, shape, scale):
    """Re-place the nominal thumbnail's tiles at their final positions.

    This mirrors Mosaic.assemble_channel at thumbnail resolution, so it
    shows where each tile actually lands in the output image.
    """
    thumb = aligner.reader.thumbnail
    md = aligner.metadata
    tile_shape = np.round(md.size * scale).astype(int)
    angle = getattr(aligner, "cycle_angle", 0)
    out = np.zeros(shape, dtype=thumb.dtype)
    for nominal, final in zip(md.positions, aligner.positions):
        crop = utils.crop(thumb, (nominal - md.origin) * scale, tile_shape)
        if crop.size:
            utils.paste(out, crop, final * scale, angle, np.fmax)
    return out


def overlay(ref, img):
    """RGB composite: cycle 0 in magenta, the other cycle in green."""
    a = to_uint8(ref)
    b = to_uint8(img)
    return np.dstack([a, b, a])


def to_uint8(img):
    img = img.astype(float)
    nonzero = img[img > 0]
    if nonzero.size == 0:
        return np.zeros(img.shape, np.uint8)
    lo, hi = np.percentile(nonzero, [1, 99.5])
    img = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
    return (img * 255).astype(np.uint8)


def save_image(path, img):
    longest = max(img.shape[:2])
    if longest > IMAGE_MAX_SIZE:
        factor = IMAGE_MAX_SIZE / longest
        img = skimage.transform.rescale(
            img, factor, channel_axis=2 if img.ndim == 3 else None,
            preserve_range=True, anti_aliasing=True,
        ).astype(np.uint8)
    Image.fromarray(img).save(path, quality=JPEG_QUALITY, optimize=True)


def model_distance_um(ea):
    """Distance of each final tile position from the linear stage model."""
    predicted = ea.lr.predict(ea.metadata.positions)
    return np.linalg.norm(ea.positions - predicted, axis=1) * ea.metadata.pixel_size


def stage_model(coef):
    """Decompose the nominal->final linear map into scale, rotation, shear.

    Columns of coef are the images of the stage y and x axes ([y, x] order).
    """
    coef = np.asarray(coef, float)
    ay, ax = coef[:, 0], coef[:, 1]
    angle_x = math.degrees(math.atan2(ax[0], ax[1]))
    angle_y = math.degrees(math.atan2(-ay[1], ay[0]))
    return {
        "coef": coef,
        "scale_y": np.linalg.norm(ay),
        "scale_x": np.linalg.norm(ax),
        "rotation_deg": angle_x,
        "shear_deg": angle_y - angle_x,
        "is_identity": bool(np.array_equal(coef, np.eye(2))),
    }


def percentiles(values, qs):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": v for q, v in zip(qs, np.percentile(values, qs))}


def write_column(path, name, values):
    write_csv(path, [name], [[v] for v in np.asarray(values).ravel()])


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([csv_value(v) for v in row])


def csv_value(v):
    v = scalar(v)
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    if isinstance(v, float):
        # Adding 0.0 turns -0.0 into 0.0.
        return f"{v + 0.0:.6g}"
    return v


def scalar(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


def clean(obj):
    """Convert numpy types to JSON-safe Python, mapping NaN/inf to None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    obj = scalar(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    return obj
