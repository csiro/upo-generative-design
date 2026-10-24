"""
Figure 1 panel: 3-D protein fitness landscape (Plotly).

Plotly uses WebGL depth-buffering so scatter points always render
correctly in front of the surface — no z-ordering hacks needed.

Run:  python upo/figure1.py
Outputs: figure1.svg, figure1.png  (saved next to this file)
"""

from pathlib import Path

import matplotlib
import numpy as np
import plotly.graph_objects as go
from scipy.stats import multinomial, multivariate_normal

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = Path(__file__).parent

# ── Mixture-model fitness landscape ───────────────────────────────────────

MUS = np.array(
    [
        [-5.0, 1.5],  # 0 – high fitness (left)
        [0.5, 2.0],  # 1 – high fitness (centre)
        [5.5, 1.5],  # 2 – high fitness (right)
        [-6.5, -1.5],  # 3 – low fitness
        [-2.5, -2.0],  # 4 – low fitness
        [2.5, -2.0],  # 5 – low fitness
        [6.5, -1.0],  # 6 – low fitness
    ]
)

# Peak 1 (centre) is the global optimum; 0 and 2 are secondary high-fitness peaks
WEIGHTS = np.array([3.6, 5.0, 2.5, 1.3, 0.7, 1.1, 0.5])
WEIGHTS /= WEIGHTS.sum()

HIGH_IDX = [0, 1, 2]
LOW_IDX = [3, 4, 5, 6]


def _rot_cov(sx, sy, deg):
    """Anisotropic covariance with std (sx, sy) rotated by `deg` degrees."""
    t = np.radians(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return R @ np.diag([sx**2, sy**2]) @ R.T


# Per-peak covariances — varied widths and orientations for an irregular
# landscape. Note: visible peak height ∝ WEIGHTS / sqrt(det(cov)), so broader
# peaks read shorter; the tight, heavy centre peak stays the global optimum.
COVS = np.array(
    [
        _rot_cov(1.7, 0.8, 30),  # 0 – elongated diagonal ridge
        _rot_cov(1.0, 1.1, 0),  # 1 – tight & tall global optimum
        _rot_cov(0.9, 1.6, -25),  # 2 – narrow, tilted
        _rot_cov(1.6, 1.2, 0),  # 3 – broad low blob
        _rot_cov(2.0, 0.7, 15),  # 4 – long shallow streak
        _rot_cov(0.8, 1.4, 45),  # 5 – compact, steeply tilted
        _rot_cov(1.4, 0.9, -40),  # 6 – modest tilted bump
    ]
)


def mpl_to_plotly_colorscale(name, n=32):
    """Convert a matplotlib colormap into a Plotly colorscale list."""
    cmap = matplotlib.colormaps[name]
    return [
        [
            i / (n - 1),
            "rgb({},{},{})".format(
                *(int(c * 255) for c in cmap(i / (n - 1))[:3])
            ),
        ]
        for i in range(n)
    ]


COLORSCALE = mpl_to_plotly_colorscale("coolwarm")


def build_norms():
    return np.array(
        [multivariate_normal(mean=m, cov=S) for m, S in zip(MUS, COVS)]
    )


def mixture_px(X, norms):
    pX = np.array([n.pdf(X) for n in norms]).squeeze()
    return WEIGHTS @ pX * 100


def sample_components(norms, component_weights, n):
    w = np.array(component_weights, dtype=float)
    w /= w.sum()
    pw = multinomial(n=1, p=w)
    z = pw.rvs(size=n).astype(bool)
    return np.array([norms[z[i]][0].rvs() for i in range(n)])


# ── Sampling ───────────────────────────────────────────────────────────────


def uhts_samples(
    norms, bounds, n=80, alpha=0.7, express_lo=0.03, express_hi=0.10
):
    """
    uHTS screening maps out the landscape, but a real library rarely samples
    the *infeasible* flats (very low fitness — proteins that don't even
    express). Acceptance is a band-pass on fitness: suppressed in the dead
    zone, highest across the expressed slopes / basins, and thinned again on
    the rare summits.
    """
    xbound, ybound = bounds
    # Global summit value as a stable normaliser (peaks are well separated)
    zmax = mixture_px(MUS, norms).max()
    kept = []
    while len(kept) < n:
        cand = np.column_stack(
            [
                np.random.uniform(-xbound, xbound, n * 8),
                np.random.uniform(-ybound, ybound, n * 8),
            ]
        )
        zn = mixture_px(cand, norms) / zmax
        # ramp up out of the infeasible dead zone
        express = np.clip((zn - express_lo) / (express_hi - express_lo), 0, 1)
        # keep summits sparse
        p_keep = express * (1.0 - alpha * zn)
        accept = np.random.uniform(size=len(cand)) < p_keep
        kept.extend(cand[accept].tolist())
    return np.array(kept[:n])


def generative_samples(
    norms,
    X_uhts,
    n=30,
    top_frac=0.10,
    reproduce_sigma=0.5,
    explore_frac=0.25,
    explore_pull=0.45,
    enrich_power=3.0,
):
    """
    VSD is trained to reproduce the top ~10% of the uHTS screen, then explore
    a little further. Generate leads from the best screened variants: jitter
    around them to reproduce, and pull an exploration fraction toward the
    nearest high-fitness peak (climbing higher than any screened point).

    Anchors are chosen with probability ∝ fitness**enrich_power, so leads are
    enriched on the most active peaks (densest on the global optimum, sparser
    on the lower peaks).
    """
    z_uhts = mixture_px(X_uhts, norms)
    n_top = max(2, int(np.ceil(len(X_uhts) * top_frac)))
    top_idx = np.argsort(z_uhts)[-n_top:]
    anchors = X_uhts[top_idx]
    anchor_p = z_uhts[top_idx] ** enrich_power
    anchor_p /= anchor_p.sum()

    high_centres = MUS[HIGH_IDX]
    leads = []
    for _ in range(n):
        base = anchors[np.random.choice(len(anchors), p=anchor_p)]
        pt = base + np.random.normal(scale=reproduce_sigma, size=2)
        if np.random.uniform() < explore_frac:
            # nudge toward the nearest high-fitness summit (extrapolate up)
            nearest = high_centres[
                np.argmin(np.linalg.norm(high_centres - pt, axis=1))
            ]
            pt = pt + explore_pull * (nearest - pt)
        leads.append(pt)

    # Cheeky "hero" lead planted right on the global optimum, with a touch of
    # jitter so it sits convincingly on the summit. Tallest peak by visible
    # height (∝ weight / sqrt(det(cov))), robust to the varied covariances.
    peak_heights = WEIGHTS / np.sqrt(np.linalg.det(COVS))
    global_opt = MUS[np.argmax(peak_heights)]
    leads.append(global_opt + np.random.normal(scale=0.15, size=2))

    return np.array(leads)


# ── Figure ─────────────────────────────────────────────────────────────────


def _sphere_mesh(points, zc, radii, color, opacity=1.0, n_u=14, n_v=9):
    """One Mesh3d holding a lit sphere at each point (radii = per-axis)."""
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v)
    unit = np.column_stack(
        [
            (np.cos(uu) * np.sin(vv)).ravel(),
            (np.sin(uu) * np.sin(vv)).ravel(),
            np.cos(vv).ravel(),
        ]
    )
    P = unit.shape[0]
    faces = []
    for a in range(n_v - 1):
        for b in range(n_u - 1):
            p0 = a * n_u + b
            faces.append([p0, p0 + 1, p0 + n_u])
            faces.append([p0 + 1, p0 + n_u + 1, p0 + n_u])
    faces = np.array(faces)

    scaled = unit * np.asarray(radii)
    xs, ys, zs, I, J, K = [], [], [], [], [], []
    for idx, (px, py) in enumerate(points):
        xs.append(px + scaled[:, 0])
        ys.append(py + scaled[:, 1])
        zs.append(zc[idx] + scaled[:, 2])
        off = idx * P
        I.append(faces[:, 0] + off)
        J.append(faces[:, 1] + off)
        K.append(faces[:, 2] + off)
    return go.Mesh3d(
        x=np.concatenate(xs),
        y=np.concatenate(ys),
        z=np.concatenate(zs),
        i=np.concatenate(I),
        j=np.concatenate(J),
        k=np.concatenate(K),
        color=color,
        opacity=opacity,
        flatshading=False,
        lighting=dict(
            ambient=0.55, diffuse=0.9, specular=0.5, roughness=0.35, fresnel=0.1
        ),
        lightposition=dict(x=150, y=200, z=400),
        showscale=False,
        hoverinfo="skip",
    )


def _legend_proxy(color, name):
    """Invisible Scatter3d so a flat-coloured swatch shows in the legend."""
    return go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode="markers",
        marker=dict(size=12, color=color),
        name=name,
        showlegend=True,
        hoverinfo="skip",
    )


def make_figure(norms):
    ngrid_x, ngrid_y = 200, 120
    xbound, ybound = 9.0, 5.0  # rectangular: wide in X to match 16:9 canvas
    xi = np.linspace(-xbound, xbound, ngrid_x)
    yi = np.linspace(-ybound, ybound, ngrid_y)
    nX, nY = np.meshgrid(xi, yi)
    grid = np.vstack([nX.ravel(), nY.ravel()]).T
    Z = mixture_px(grid, norms).reshape(ngrid_y, ngrid_x)

    X_uhts = uhts_samples(norms, bounds=(xbound, ybound))
    X_gen = generative_samples(norms, X_uhts)
    Z_uhts = mixture_px(X_uhts, norms)
    Z_gen = mixture_px(X_gen, norms)

    # Per-axis sphere radii that compensate the manual aspect ratio
    # (x=2.0, y=1.0, z=0.55) so the spheres render round on screen.
    r_vis = 0.018  # sphere radius as a fraction of the aspect-box
    zrange = float(Z.max() - Z.min())
    radii = (
        r_vis * (2 * xbound) / 2.0,
        r_vis * (2 * ybound) / 1.0,
        r_vis * zrange / 0.55,
    )
    # Rest the spheres on the surface (centre = local height + one radius)
    clear = Z.max() * 0.01
    zc_uhts = Z_uhts + radii[2] + clear
    zc_gen = Z_gen + radii[2] + clear

    fig = go.Figure()

    # ── Surface ───────────────────────────────────────────────────────────
    fig.add_trace(
        go.Surface(
            x=nX,
            y=nY,
            z=Z,
            colorscale=COLORSCALE,
            cmin=Z.min(),
            cmax=Z.max(),
            showscale=True,
            colorbar=dict(
                thickness=24,
                len=0.45,
                x=0.97,
                y=0.55,
                tickvals=[Z.min(), Z.max()],
                ticktext=["Low", "High"],
                tickfont=dict(size=26),
                title=dict(text="Fitness", font=dict(size=26), side="right"),
                outlinewidth=0.5,
            ),
            lighting=dict(
                ambient=0.85, diffuse=0.5, roughness=0.5, specular=0.05
            ),
            lightposition=dict(x=1, y=1, z=2),
            name="Fitness landscape",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # ── uHTS variants – orange spheres ────────────────────────────────────
    fig.add_trace(_sphere_mesh(X_uhts, zc_uhts, radii, "#C97B4A"))
    fig.add_trace(_legend_proxy("#C97B4A", "uHTS screened variants"))

    # ── VSD generated leads – blue spheres ────────────────────────────────
    fig.add_trace(_sphere_mesh(X_gen, zc_gen, radii, "#42658A"))
    fig.add_trace(_legend_proxy("#42658A", "VSD generated leads"))

    # ── Layout ────────────────────────────────────────────────────────────
    axis_style = dict(
        showticklabels=False,
        showgrid=False,
        showbackground=False,
        zeroline=False,
        showline=False,
        ticks="",
    )

    title_font = dict(size=22)
    fig.update_layout(
        width=1600,
        height=900,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        scene=dict(
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
            # x-axis label at the bottom front edge; y-axis label removed
            # (becomes a depth axis from the front-on view — would clash);
            # z-axis label on the left vertical edge
            xaxis=dict(
                title=dict(text="Sequence space", font=title_font), **axis_style
            ),
            yaxis=dict(title=dict(text="", font=title_font), **axis_style),
            zaxis=dict(
                title=dict(text="Fitness", font=title_font), **axis_style
            ),
            aspectmode="manual",
            aspectratio=dict(x=2.0, y=1.0, z=0.55),
        ),
        scene_camera=dict(
            # rotated steeper (more top-down) at the same distance as before
            # so far slopes are visible; |eye| ≈ 1.53 unchanged, x=0 → no twist
            eye=dict(x=0.0, y=-0.55, z=1.43),
            up=dict(x=0, y=0, z=1),
        ),
        # Legend: top-right corner, away from the z-axis label (left) and
        # x-axis label (bottom centre)
        legend=dict(
            x=0.68,
            y=0.97,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=26),
        ),
    )

    return fig


def save_fig(fig, name):
    for ext in ("svg", "png"):
        path = OUTPUT_DIR / f"{name}.{ext}"
        # scale=3 → 4800×2700 px PNG, suitable for 300 DPI print at 16×9 in
        fig.write_image(str(path), scale=3 if ext == "png" else 1)
        print(f"  saved {path}")


# ── Entry point ────────────────────────────────────────────────────────────


def main():
    norms = build_norms()
    fig = make_figure(norms)
    save_fig(fig, "figure1")


if __name__ == "__main__":
    main()
