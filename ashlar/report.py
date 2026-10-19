import copy
import fpdf
import functools
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
import seaborn as sns
import skimage.exposure
import skimage.transform
import sklearn.neighbors

from . import qc


cmap_yellow = mcolors.LinearSegmentedColormap(
    name="a_yellow",
    segmentdata={
        'red': ((0, 0.1, 0.1), (1, 1, 1)),
        'green': ((0, 0.1, 0.1), (1, 1, 1)),
        'blue': ((0, 0, 0), (1, 0.8, 0.8)),
    },
)


def generate_report(path, aligners, summary=None):
    aligner0, *aligners = aligners
    if summary is None:
        summary = qc.summarize([aligner0, *aligners])
    summary = qc.clean(summary)
    pdf = PDF(unit="pt", format="letter")
    pdf.set_font("Helvetica")

    pdf.add_page()
    add_summary(pdf, summary)

    pdf.add_page()
    fig = plot_edge_map(aligner0, aligner0.reader.thumbnail)
    fig.set_size_inches(7.5, 8)
    pdf.add_figure(fig, "Cycle 1: Tile pair alignment map")
    plt.close(fig)
    add_legend(pdf, "edge_map")

    pdf.add_page()
    fig = plot_edge_quality(aligner0)
    fig.set_size_inches(7.5, 7.5)
    pdf.add_figure(fig, "Cycle 1: Tile pair alignment quality")
    plt.close(fig)
    add_legend(pdf, "edge_quality")

    if hasattr(aligner0, "foreground_energy"):
        pdf.add_page()
        fig = plot_foreground(aligner0, aligner0.reader.thumbnail)
        fig.set_size_inches(7.5, 8)
        pdf.add_figure(fig, "Cycle 1: Foreground tiles")
        plt.close(fig)
        add_legend(pdf, "foreground")

    scale = aligner0.reader.thumbnail_scale
    shape = np.ceil(np.array(aligner0.mosaic_shape) * scale).astype(int)
    ref_thumb = qc.registered_thumbnail(aligner0, shape, scale)
    for i, aligner in enumerate(aligners, 2):
        pdf.add_page()
        fig = plot_layer_map(aligner, aligner.reader.thumbnail)
        fig.set_size_inches(7.5, 8)
        pdf.add_figure(fig, f"Cycle {i}: Cycle alignment map")
        plt.close(fig)
        add_legend(pdf, "layer_map")

        pdf.add_page()
        thumb = qc.registered_thumbnail(aligner, shape, scale)
        fig = plot_overlay(ref_thumb, thumb)
        fig.set_size_inches(7.5, 8)
        pdf.add_figure(fig, f"Cycle {i}: Registered overlay on cycle 1")
        plt.close(fig)
        add_legend(pdf, "overlay")

        if getattr(aligner, "angles", None) is not None:
            pdf.add_page()
            fig = plot_layer_angles(aligner, aligner.reader.thumbnail)
            fig.set_size_inches(7.5, 8)
            pdf.add_figure(fig, f"Cycle {i}: Per-tile rotation")
            plt.close(fig)
            add_legend(pdf, "angles")

    pdf.output(path)


def pdf_text(text):
    """Fold text into the core PDF fonts' Latin-1 range (Ashlar warns with Greek mu)."""
    text = str(text).replace("\u03bc", "\u00b5")
    return text.encode("latin-1", "replace").decode("latin-1")


def fmt(v, spec=".3g"):
    if v is None:
        return "-"
    if isinstance(v, float):
        return format(v, spec)
    return str(v)


def add_summary(pdf, summary):
    st = summary["stitching"]
    pdf.add_figure_title("ASHLAR stitching and registration QC")
    with pdf.local_context(font_size=9):
        pdf.ln(4)
        pdf.multi_cell(
            w=0,
            text=(
                f"ashlar {summary['ashlar_version']}. Cycle numbers start at"
                " 1 here (qc.json and the CSV files count from 0). Lengths in"
                " \xb5m unless stated. Errors are -log NCC (lower is better)."
            ),
        )
        pdf.ln(4)
        sm = st["stage_model"]
        rows = [
            ("Tiles", fmt(st["n_tiles"])),
            ("Pixel size (\xb5m)", fmt(summary["pixel_size_um"], ".4g")),
            ("Mosaic shape (px)", " x ".join(map(str, summary["mosaic_shape"]))),
            ("Neighbour pairs", fmt(st["n_edges"])),
            ("  accepted", fmt(st["n_edges_accepted"])),
            ("  rejected: error above threshold", fmt(st["n_edges_rejected_error"])),
            ("  rejected: shift above max", fmt(st["n_edges_rejected_shift"])),
            ("Error threshold (null p%g)" % (st["alpha"] * 100), fmt(st["max_error"])),
            ("Pair error p50 / null p50",
             f"{fmt(st['edge_error_percentiles']['p50'])} / "
             f"{fmt(st['null_error_percentiles']['p50'])}"),
            ("Pairs no better than random", fmt(st["edges_like_null_fraction"], ".0%")),
            ("Spanning tree components", fmt(st["n_components"])),
            ("Tiles outside largest component",
             fmt(st["n_tiles_outside_largest_component"])),
            ("Stage scale y / x",
             f"{fmt(sm['scale_y'], '.4f')} / {fmt(sm['scale_x'], '.4f')}"),
            ("Stage rotation / shear (deg)",
             f"{fmt(sm['rotation_deg'], '.3f')} / {fmt(sm['shear_deg'], '.3f')}"),
            ("Tile residual from stage model p50 / p95",
             f"{fmt(st['model_residual_um_percentiles']['p50'])} / "
             f"{fmt(st['model_residual_um_percentiles']['p95'])}"),
            ("Foreground tiles", fmt(st["n_foreground_tiles"])),
        ]
        add_table(pdf, ("Cycle 1 stitching", ""), rows, (3, 2))
        if summary["registration"]:
            pdf.ln(8)
            header = (
                "Cycle", "Offset y, x", "Angle (deg)", "Tile angle MAD",
                "Discarded (err/bg/shift)", "Kept like null",
                "Residual p50 / p95",
            )
            rows = []
            for r in summary["registration"]:
                off = r["cycle_offset_um"]
                res = r["residual_um_percentiles"]
                rows.append((
                    str(r["cycle"] + 1),
                    f"{fmt(off[0], '.1f')}, {fmt(off[1], '.1f')}",
                    fmt(r["cycle_angle"], ".2f"),
                    fmt(r["tile_angle_mad"], ".2f"),
                    f"{r['n_discard']}/{r['n_tiles']} "
                    f"({r['n_discard_error']}/{r['n_discard_background']}/"
                    f"{r['n_discard_shift']})",
                    fmt(r["n_kept_like_null"]),
                    f"{fmt(res['p50'])} / {fmt(res['p95'])}",
                ))
            add_table(pdf, header, rows, (1, 2, 1.5, 1.5, 2.5, 1.5, 2))
        if summary["warnings"]:
            pdf.ln(8)
            with pdf.local_context(font_style="B"):
                pdf.cell(text="Warnings", new_x="LEFT", new_y="NEXT")
            for w in summary["warnings"]:
                pdf.multi_cell(w=0, text=pdf_text(f"- {w}"))
        pdf.ln(6)
        add_legend(pdf, "summary")


def add_table(pdf, header, rows, widths):
    with pdf.table(col_widths=widths, text_align="LEFT", line_height=11) as t:
        row = t.row()
        for h in header:
            row.cell(h)
        for r in rows:
            row = t.row()
            for v in r:
                row.cell(v)


LEGENDS = {
    "summary": (
        "Offsets are the coarse cycle-to-cycle translation from thumbnail"
        " correlation. Discarded tiles were placed by the stage model:"
        " err = registration failed, bg = reference tile is background,"
        " shift = registered too far from the model. 'Kept like null' counts"
        " kept tiles whose error is no better than the cycle 1 threshold, so"
        " their registration may be spurious. Residual is each usable tile's"
        " distance from the rigid model prediction."
    ),
    "foreground": (
        "Tiles coloured by image energy (norm of the Laplacian-whitened"
        " alignment channel). Tiles below the Otsu threshold (inset, dashed"
        " line) count as background: they are skipped for rotation"
        " refinement and their later-cycle registrations are discarded."
        " Otsu always splits the tiles in two, so on a slide that is all"
        " tissue some real tissue tiles are marked background."
    ),
    "overlay": (
        "Cycle 1 (magenta) and this cycle (green) at their final registered"
        " positions, built from the thumbnails. Well-registered tissue is"
        " white or grey; colour fringes or doubled structures mean"
        " misregistration. Black areas lie outside this cycle's coverage."
    ),
    "angles": (
        "Residual rotation of each tile against its cycle 1 reference after"
        " the coarse cycle rotation, at nominal positions (left), and their"
        " distribution (right). The cycle rotation is refined by the median"
        " shown as the dashed line. A wide spread or spatial trend suggests"
        " non-rigid deformation that a single rotation cannot correct."
        " Background tiles are not measured."
    ),
    "edge_map": (
        "Each node is a tile at its nominal stage position, drawn over the"
        " cycle 1 thumbnail. Lines join overlapping neighbours, coloured by"
        " alignment error (-log NCC; darker blue is worse). Dark grey lines"
        " were rejected (error above the permutation threshold or shift above"
        " --maximum-shift). Solid lines form the spanning tree used to place"
        " tiles; dotted lines were not used. Islands of tiles with no solid"
        " lines were placed by the linear stage model only."
    ),
    "edge_quality": (
        "Each point is one neighbouring tile pair: alignment error against"
        " the corrective shift found. The salmon curve is the error of"
        " random non-overlapping image strips (the null distribution); the"
        " vertical dotted line is the error threshold taken from it"
        " (--stitch-alpha) and the horizontal one is --maximum-shift. Good"
        " data has most pairs well left of the salmon curve. Pairs in the"
        " salmon region are usually background-only overlaps."
    ),
    "layer_map": (
        "Each node is a tile of this cycle at its nominal position over this"
        " cycle's thumbnail, coloured by its registration shift to cycle 1."
        " Ticks show shift direction. Dark grey nodes were discarded and"
        " placed by the model: squares had a background reference tile,"
        " diamonds failed to register, grey circles shifted more than"
        " --maximum-shift from the model prediction. The dashed outline is"
        " the cycle 1 mosaic placed by the coarse (thumbnail) alignment; if"
        " it does not sit on the tissue, the coarse alignment failed."
    ),
}


def add_legend(pdf, key):
    with pdf.local_context(font_size=9):
        pdf.ln(6)
        pdf.multi_cell(w=0, text=LEGENDS[key])


class PDF(fpdf.FPDF):

    def add_figure_title(self, txt):
        with self.local_context(font_size=16, font_style="B"):
            self.cell(text=txt, w=0, align="C", new_x="LEFT", new_y="NEXT")

    def add_figure(self, fig, title):
        self.add_figure_title(title)
        if fig.findobj(mimage.AxesImage):
            fig.set_dpi(200)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            rgb = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
            # JPEG keeps image-heavy pages to a few hundred kB each.
            img = io.BytesIO()
            rgb.save(img, format="JPEG", quality=85)
        else:
            img = io.BytesIO()
            fig.savefig(img, format="svg")
        self.image(img, w=self.epw)


def plot_edge_map(
    aligner,
    img=None,
    pos="metadata",
    cmap=None,
    width=None,
    node_size=None,
    font_size=None,
    im_kwargs=None,
    nx_kwargs=None,
):
    """Plot neighbor graph colored by edge alignment quality"""

    if pos == "metadata":
        centers = aligner.metadata.centers - aligner.metadata.origin
    elif pos == "aligner":
        centers = aligner.centers
    else:
        raise ValueError("pos must be either 'metadata' or 'aligner'")
    cmap = cmap or cmap_yellow
    im_kwargs = im_kwargs or {}
    nx_kwargs = nx_kwargs or {}
    fig, ax = plt.subplots()
    draw_mosaic_image(ax, aligner, img, cmap=cmap, **im_kwargs)
    error = np.array([
        aligner._cache[tuple(sorted(e))][1]
        for e in aligner.neighbors_graph.edges
    ])
    efinite = error[error < np.inf]
    emin, emax = (efinite.min(), efinite.max()) if len(efinite) > 0 else (0, 0)
    in_tree = [
        e in aligner.spanning_tree.edges
        for e in aligner.neighbors_graph.edges
    ]
    diameter = nx.diameter(aligner.neighbors_graph)
    drange = [10, 60]
    interp = functools.partial(np.interp, diameter, [10, 60])
    width = width or interp([4, 1])
    node_size = node_size or interp([100, 8])
    font_size = font_size or interp([6, 2])
    width = np.where(in_tree, width, width * 0.75)
    style = np.where(in_tree, "solid", "dotted")
    edge_cmap = mcm.Blues.with_extremes(over="#333333")
    g = aligner.neighbors_graph
    pos = np.fliplr(centers)
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color="silver",
        node_size=node_size,
        edgecolors=None,
        **nx_kwargs,
    )
    ec = nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        edge_color=error,
        edge_vmin=emin,
        edge_vmax=emax,
        edge_cmap=edge_cmap,
        width=width,
        style=style,
        **nx_kwargs,
    )
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=font_size, **nx_kwargs)
    draw_borders(ax, aligner, pos)
    cbar = fig.colorbar(
        mcm.ScalarMappable(mcolors.Normalize(emin, emax), edge_cmap),
        extend="max",
        label="Error (-log NCC)",
        location="right",
        shrink=0.5,
        ax=ax,
    )
    ax.set_frame_on(False)
    ax.margins(0)
    fig.tight_layout()
    return fig


def plot_edge_shifts(
        aligner,
        img=None,
        pos="metadata",
        bounds=True,
        all_edges=False,
        cmap=None,
        im_kwargs=None,
):
    if pos == "metadata":
        centers = aligner.metadata.centers - aligner.metadata.origin
    elif pos == "aligner":
        centers = aligner.centers
    else:
        raise ValueError("pos must be either 'metadata' or 'aligner'")
    cmap = cmap or cmap_yellow
    im_kwargs = im_kwargs or {}
    fig, ax = plt.subplots()
    draw_mosaic_image(ax, aligner, img, cmap=cmap, **im_kwargs)
    pos = np.fliplr(centers)
    # Compute per-edge relative shifts from tile positions.
    graph = aligner.neighbors_graph if all_edges else aligner.spanning_tree
    edges = np.array(list(graph.edges))
    pixel_size = aligner.reader.metadata.pixel_size
    dist = (aligner.metadata.positions - aligner.positions) * pixel_size
    shifts = dist[edges[:, 0]] - dist[edges[:, 1]]
    shift_distances = np.linalg.norm(shifts, axis=1)
    dmin = shift_distances.min()
    dmax = shift_distances.max()
    edge_cmap = mcm.Blues_r
    # Spanning tree with nodes at new tile positions, edges colored by shift
    # distance (brighter = farther).
    nx.draw(
        graph, ax=ax, with_labels=True, pos=pos, edge_color=shift_distances,
        edge_cmap=edge_cmap, width=2, node_size=100, font_size=6
    )
    if bounds:
        draw_borders(ax, aligner, pos)
    cbar = fig.colorbar(
        mcm.ScalarMappable(mcolors.Normalize(dmin, dmax), edge_cmap),
        label="Shift distance (\u00B5m)",
        location="right",
        shrink=0.5,
        ax=ax,
    )
    ax.set_frame_on(False)
    ax.margins(0)
    fig.tight_layout()
    return fig


def plot_edge_quality(aligner, annotate=True):
    xdata = np.clip(aligner.all_errors, 0, 10)
    ydata = (
        np.linalg.norm([v[0] for v in aligner._cache.values()], axis=1)
        .clip(0.01, np.inf)
        * aligner.metadata.pixel_size
    )
    pdata = np.clip(aligner.errors_negative_sampled, 0, 10)
    if len(xdata) >= 10:
        # Identify points in sparse regions of figure space by computing whether
        # each point's 5th nearest neighbor is more then 2% of the figure width
        # away. (Using k=5 allows clusters of a few points to be considered sparse)
        dens_data = np.vstack([xdata, np.log(ydata)]).T
        dens_data /= np.linalg.norm(np.ptp(dens_data, axis=0))
        tree = sklearn.neighbors.KDTree(dens_data)
        adata = tree.query(dens_data, k=5)[0][:, -1] > 0.02
    else:
        # With few enough points, annotate them all. (nearest neighbor query
        # fails with fewer than k points, anyway)
        adata = np.ones(len(xdata), bool)
    g = sns.JointGrid(x=xdata, y=ydata)
    g.plot_joint(sns.scatterplot, alpha=0.5, edgecolor="none", ax=g.ax_joint)
    histplot = functools.partial(
        sns.histplot,
        bins=40,
        stat="density",
        kde=True,
        color="tab:blue",
        ec="white",
    )
    histplot(x=xdata, ax=g.ax_marg_x)
    histplot(y=ydata, ax=g.ax_marg_y, log_scale=True)
    sns.kdeplot(x=pdata, ax=g.ax_marg_x, color="salmon")
    g.ax_joint.axvline(aligner.max_error, c="k", ls=":")
    g.ax_marg_x.axvline(aligner.max_error, c="k", ls=":")
    g.ax_joint.axhline(aligner.max_shift, c="k", ls=":")
    g.ax_marg_y.axhline(aligner.max_shift, c="k", ls=":")
    g.set_axis_labels("Error (-log NCC)", "Shift distance (\u00B5m)")
    if annotate:
        for pair, x, y, a in zip(
            aligner.neighbors_graph.edges, xdata, ydata, adata
        ):
            if a:
                g.ax_joint.annotate(str(pair), (x, y), alpha=0.5, size=6)
    g.figure.tight_layout()
    return g.figure


def plot_layer_shifts(aligner, img=None, im_kwargs=None):
    if im_kwargs is None:
        im_kwargs = {}
    fig = plt.figure()
    ax = plt.gca()
    draw_mosaic_image(ax, aligner, img, **im_kwargs)
    h, w = aligner.metadata.size
    # Bounding boxes denoting new tile positions.
    for xy in np.fliplr(aligner.positions):
        rect = mpatches.Rectangle(xy, w, h, color='black', fill=False, lw=0.5)
        ax.add_patch(rect)
    # Neighbor graph with edges hidden, i.e. just show nodes.
    nx.draw(
        aligner.neighbors_graph, ax=ax, with_labels=True,
        pos=np.fliplr(aligner.centers), edge_color='none',
        node_size=100, font_size=6
    )
    fig.set_facecolor('black')


def plot_layer_map(
    aligner,
    img=None,
    pos="metadata",
    cmap=None,
    width=None,
    node_size=None,
    font_size=None,
    im_kwargs=None,
    nx_kwargs=None,
):
    """Plot tile shift distance and direction"""

    if pos == "metadata":
        centers = aligner.metadata.centers - aligner.metadata.origin
    elif pos == "aligner":
        centers = aligner.centers
    else:
        raise ValueError("pos must be either 'metadata' or 'aligner'")
    cmap = cmap or cmap_yellow
    im_kwargs = im_kwargs or {}
    nx_kwargs = nx_kwargs or {}
    fig, ax = plt.subplots()
    draw_mosaic_image(ax, aligner, img, cmap=cmap, **im_kwargs)

    pixel_size = aligner.reader.metadata.pixel_size
    shifts = np.linalg.norm(aligner.shifts, axis=1) * pixel_size
    skeep = shifts[~aligner.discard]
    smin, smax = (skeep.min(), skeep.max()) if len(skeep) > 0 else (0, 0)
    # Map discards to the "over" color in the cmap.
    shifts[aligner.discard] = smax + 1
    # Reorder to match the graph's internal node ordering.
    node_values = shifts[np.array(aligner.neighbors_graph)]

    diameter = nx.diameter(aligner.neighbors_graph)
    drange = [10, 60]
    interp = functools.partial(np.interp, diameter, drange)
    node_size = node_size or interp([100, 8])
    font_size = font_size or interp([6, 2])
    node_cmap = mcm.Greens.with_extremes(over="#333333")
    g = aligner.neighbors_graph
    pos = np.fliplr(centers)
    qlen = np.repeat(np.min(aligner.metadata.size) * 0.45, len(pos))
    qlen[aligner.discard_background | aligner.discard_error] = 0
    q_angles = np.rad2deg(np.arctan2(*(aligner.shifts * [-1, 1]).T))
    # Outline of the reference (cycle 1) mosaic mapped into this cycle's frame
    # by inverting the coarse cycle transform from LayerAligner.coarse_align.
    ref_md = aligner.reference_aligner.metadata
    ref_outer = np.max(ref_md.positions + ref_md.size, axis=0)
    (y0, x0), (y1, x1) = ref_md.origin, ref_outer
    ref_corners = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
    mcenter = aligner.metadata.center
    tform = skimage.transform.AffineTransform(
        rotation=np.deg2rad(aligner.cycle_angle)
    )
    corners = (
        tform.inverse(ref_corners - mcenter - aligner.cycle_offset)
        + mcenter
        - aligner.metadata.origin
    )
    ax.add_patch(
        mpatches.Polygon(
            np.fliplr(corners),
            closed=True,
            color=mcm.Blues(0.6),
            lw=2,
            linestyle="--",
            fill=False,
        )
    )
    ax.quiver(
        pos[:, 0],
        pos[:, 1],
        qlen,
        [0] * len(pos),
        shifts,
        cmap=node_cmap,
        clim=(smin, smax),
        angles=q_angles,
        scale_units="x",
        scale=1,
        headwidth=1,
        headlength=1,
        headaxislength=1,
    )

    def subgraph(cond):
        return g.subgraph(np.nonzero(cond)[0])

    def draw_nodes(cond, **kwargs):
        nx.draw_networkx_nodes(
            subgraph(cond),
            pos=pos,
            ax=ax,
            cmap=node_cmap,
            vmin=smin,
            vmax=smax,
            node_color=node_values[cond[np.array(aligner.neighbors_graph)]],
            node_size=node_size,
            edgecolors=None,
            **nx_kwargs,
            **kwargs,
        )

    def draw_labels(cond, **kwargs):
        nx.draw_networkx_labels(
            subgraph(cond),
            pos=pos,
            ax=ax,
            font_size=font_size,
            **nx_kwargs,
            **kwargs,
        )

    draw_nodes(aligner.discard_background, node_shape="s")
    draw_nodes(
        aligner.discard_error & ~aligner.discard_background, node_shape="D"
    )
    draw_nodes(
        ~(aligner.discard_background | aligner.discard_error), node_shape="o"
    )
    draw_labels(aligner.discard, font_color="gray")
    draw_labels(~aligner.discard, font_color="k")
    draw_borders(ax, aligner, pos)

    cbar = fig.colorbar(
        mcm.ScalarMappable(mcolors.Normalize(smin, smax), node_cmap),
        extend="max",
        label="Shift (\xb5m)",
        location="right",
        shrink=0.5,
        ax=ax,
    )

    ax.set_frame_on(False)
    ax.margins(0)
    fig.tight_layout()

    return fig


def plot_foreground(aligner, img=None, cmap=None, im_kwargs=None):
    """Plot tiles coloured by foreground energy with the Otsu threshold."""
    centers = aligner.metadata.centers - aligner.metadata.origin
    cmap = cmap or cmap_yellow
    im_kwargs = im_kwargs or {}
    fig, ax = plt.subplots()
    draw_mosaic_image(ax, aligner, img, cmap=cmap, **im_kwargs)
    energy = aligner.foreground_energy
    threshold = aligner.foreground_threshold
    node_cmap = mcm.RdBu_r
    norm = mcolors.TwoSlopeNorm(
        vcenter=threshold,
        vmin=min(energy.min(), threshold - 1e-9),
        vmax=max(energy.max(), threshold + 1e-9),
    )
    pos = np.fliplr(centers)
    diameter = nx.diameter(aligner.neighbors_graph)
    interp = functools.partial(np.interp, diameter, [10, 60])
    for fg, marker in ((True, "o"), (False, "s")):
        sel = aligner.foreground == fg
        ax.scatter(
            *pos[sel].T, c=energy[sel], cmap=node_cmap, norm=norm,
            s=interp([100, 8]), marker=marker, edgecolors="k", linewidths=0.3,
        )
    draw_borders(ax, aligner, pos)
    fig.colorbar(
        mcm.ScalarMappable(norm, node_cmap), label="Energy", location="right",
        shrink=0.5, ax=ax,
    )
    inset = ax.inset_axes([0.0, -0.22, 0.4, 0.16])
    inset.hist(energy, bins=30, color="tab:gray")
    inset.axvline(threshold, c="k", ls="--")
    inset.set_xlabel("Energy", fontsize=7)
    inset.tick_params(labelsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.margins(0)
    fig.tight_layout()
    return fig


def plot_overlay(ref, img):
    """Show the magenta/green overlay of two registered thumbnails."""
    fig, ax = plt.subplots()
    ax.imshow(qc.overlay(ref, img), interpolation="none")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_layer_angles(aligner, img=None, cmap=None, im_kwargs=None):
    """Plot per-tile residual rotation angles as a map and histogram."""
    centers = aligner.metadata.centers - aligner.metadata.origin
    cmap = cmap or cmap_yellow
    im_kwargs = im_kwargs or {}
    angles = aligner.angles
    finite = np.isfinite(angles)
    median = np.nanmedian(angles) if finite.any() else 0
    fig, (ax, hax) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [3, 1]}
    )
    draw_mosaic_image(ax, aligner, img, cmap=cmap, **im_kwargs)
    pos = np.fliplr(centers)
    lim = max(np.nanmax(np.abs(angles - median)) if finite.any() else 0, 0.05)
    node_cmap = mcm.PuOr
    norm = mcolors.Normalize(median - lim, median + lim)
    diameter = nx.diameter(aligner.neighbors_graph)
    size = np.interp(diameter, [10, 60], [100, 8])
    ax.scatter(
        *pos[finite].T, c=angles[finite], cmap=node_cmap, norm=norm, s=size,
        edgecolors="k", linewidths=0.3,
    )
    ax.scatter(*pos[~finite].T, c="#333333", s=size * 0.5, marker="s")
    draw_borders(ax, aligner, pos)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.margins(0)
    fig.colorbar(
        mcm.ScalarMappable(norm, node_cmap), label="Residual angle (deg)",
        location="bottom", shrink=0.6, ax=ax,
    )
    hax.hist(angles[finite], bins=30, orientation="horizontal", color="tab:gray")
    hax.axhline(median, c="k", ls="--")
    hax.set_ylabel("Residual angle (deg)")
    hax.set_title(
        f"cycle angle {aligner.cycle_angle:.2f}\n"
        f"(coarse {getattr(aligner, 'cycle_angle_coarse', np.nan):.2f})",
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def draw_mosaic_image(ax, aligner, img, **kwargs):
    if img is None:
        img = [[0]]
    cmax = np.max(aligner.metadata.centers - aligner.metadata.origin, axis=0)
    h, w = cmax + aligner.metadata.size / 2
    if "vmin" not in kwargs:
        kwargs["vmin"] = np.percentile(img, 1)
    if "vmax" not in kwargs:
        kwargs["vmax"] = np.percentile(img, 99)
    ax.imshow(img, extent=(-0.5, w-0.5, h-0.5, -0.5), **kwargs)


def draw_borders(ax, aligner, pos):
    rh, rw = aligner.metadata.size
    for x, y in pos - (rw / 2, rh / 2):
        rect = mpatches.Rectangle(
            (x, y),
            rw,
            rh,
            color=(0.2, 0.2, 0.2),
            fill=False,
            lw=0.25,
            zorder=0.5,
        )
        ax.add_patch(rect)
