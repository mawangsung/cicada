"""Plotly 3D Bloch sphere visualization for ODIN QuantumLogicGate v2.0.

Renders single qubits and the full Huginn-Muninn-Metin 3-qubit register
as interactive Bloch spheres. Entanglement entropy > 0.1 draws connection
lines between qubit spheres.
"""

import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..state.qubit import Qubit
from ..state.register import QuantumRegister

# Norse qubit labels
QUBIT_LABELS = {0: "Huginn", 1: "Muninn", 2: "Metin"}
QUBIT_COLORS = {0: "#4a90d9", 1: "#e8a838", 2: "#5cb85c"}
ENTANGLEMENT_COLOR = "#c44dff"


def _sphere_mesh(n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, z) meshgrid for a unit sphere surface."""
    theta = np.linspace(0, math.pi, n)
    phi = np.linspace(0, 2 * math.pi, n)
    T, P = np.meshgrid(theta, phi)
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    return x, y, z


def _axis_traces(label_offset: float = 1.25) -> list[go.Scatter3d]:
    """Return x/y/z axis line traces for a Bloch sphere."""
    traces = []
    axes = [
        ([0, 1.1], [0, 0], [0, 0], "+X"),
        ([-1.1, 0], [0, 0], [0, 0], "-X"),
        ([0, 0], [0, 1.1], [0, 0], "+Y"),
        ([0, 0], [-1.1, 0], [0, 0], "-Y"),
        ([0, 0], [0, 0], [0, 1.1], "|0⟩"),
        ([0, 0], [0, 0], [-1.1, 0], "|1⟩"),
    ]
    for xs, ys, zs, text in axes:
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines+text",
            line=dict(color="gray", width=2),
            text=["", text],
            textposition="top center",
            showlegend=False,
        ))
    return traces


def _sphere_surface(color: str, opacity: float = 0.12) -> go.Surface:
    x, y, z = _sphere_mesh()
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        opacity=opacity,
        showscale=False,
        hoverinfo="skip",
    )


def _state_arrow(bv: tuple[float, float, float], color: str, label: str) -> list[go.Scatter3d]:
    """Return a state-vector arrow (line + tip marker) for Bloch vector bv."""
    x, y, z = bv
    # Shaft
    shaft = go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode="lines",
        line=dict(color=color, width=6),
        showlegend=False,
        hovertemplate=f"{label}<br>x={x:.3f}, y={y:.3f}, z={z:.3f}<extra></extra>",
    )
    # Tip
    tip = go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode="markers+text",
        marker=dict(size=8, color=color, symbol="circle"),
        text=[label],
        textposition="top center",
        showlegend=False,
    )
    return [shaft, tip]


class BlochSphere:
    """Plotly-based Bloch sphere renderer."""

    # ── Single qubit ────────────────────────────────────────────────────────

    @staticmethod
    def render_single(qubit: Qubit, label: str = "q", show: bool = True) -> go.Figure:
        """Render a single qubit state on a Bloch sphere."""
        bv = qubit.bloch_vector()
        color = "#4a90d9"

        traces: list = [_sphere_surface(color)]
        traces += _axis_traces()
        traces += _state_arrow(bv, color, label)

        fig = go.Figure(data=traces)
        theta, phi = qubit.bloch_angles()
        p0 = qubit.probability_zero()
        fig.update_layout(
            title=dict(
                text=(f"Bloch Sphere — {label}<br>"
                      f"<sup>θ={math.degrees(theta):.1f}°  φ={math.degrees(phi):.1f}°  "
                      f"P(0)={p0:.3f}  P(1)={1-p0:.3f}</sup>"),
                x=0.5,
            ),
            scene=dict(
                xaxis=dict(range=[-1.4, 1.4], title="X"),
                yaxis=dict(range=[-1.4, 1.4], title="Y"),
                zaxis=dict(range=[-1.4, 1.4], title="Z"),
                dragmode="orbit",
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=60),
            width=600,
            height=600,
        )
        if show:
            fig.show()
        return fig

    # ── 3-qubit register ─────────────────────────────────────────────────────

    @staticmethod
    def render_register(register: QuantumRegister, show: bool = True) -> go.Figure:
        """Render Huginn, Muninn, Metin on three side-by-side Bloch spheres."""
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "scatter3d"}] * 3],
            subplot_titles=["Huginn (q0)", "Muninn (q1)", "Metin (q2)"],
            horizontal_spacing=0.02,
        )

        entropies = [register.entanglement_entropy(i) for i in range(3)]

        for i in range(3):
            bv = register.bloch_vector(i)
            color = QUBIT_COLORS[i]
            col = i + 1
            scene = f"scene{'' if col == 1 else col}"

            # Sphere surface
            fig.add_trace(_sphere_surface(color), row=1, col=col)

            # Axes
            for t in _axis_traces():
                fig.add_trace(t, row=1, col=col)

            # State arrow
            for t in _state_arrow(bv, color, QUBIT_LABELS[i]):
                fig.add_trace(t, row=1, col=col)

        # Update scenes
        scene_kwargs = dict(
            xaxis=dict(range=[-1.4, 1.4], title=""),
            yaxis=dict(range=[-1.4, 1.4], title=""),
            zaxis=dict(range=[-1.4, 1.4], title=""),
            dragmode="orbit",
            aspectmode="cube",
        )
        fig.update_layout(
            scene=scene_kwargs,
            scene2=scene_kwargs,
            scene3=scene_kwargs,
            title=dict(
                text=(
                    "ODIN QuantumLogicGate v2.0 — Huginn · Muninn · Metin<br>"
                    f"<sup>S(Huginn)={entropies[0]:.3f}  "
                    f"S(Muninn)={entropies[1]:.3f}  "
                    f"S(Metin)={entropies[2]:.3f}</sup>"
                ),
                x=0.5,
            ),
            margin=dict(l=0, r=0, b=0, t=80),
            width=1200,
            height=500,
        )
        if show:
            fig.show()
        return fig

    # ── Entanglement visualization ───────────────────────────────────────────

    @staticmethod
    def render_entanglement(register: QuantumRegister, show: bool = True) -> go.Figure:
        """Render Bloch spheres with rune annotations and entanglement indicators.

        Qubit spheres are spaced along the X axis.
        Entanglement entropy > 0.1 is shown as a glowing arc between spheres.
        """
        offsets = {0: (-4.0, 0, 0), 1: (0.0, 0, 0), 2: (4.0, 0, 0)}
        entropies = [register.entanglement_entropy(i) for i in range(3)]
        state_dict = register.to_dict()

        traces = []

        for i in range(3):
            ox, oy, oz = offsets[i]
            bv = register.bloch_vector(i)
            color = QUBIT_COLORS[i]

            # Sphere mesh
            sx, sy, sz = _sphere_mesh(30)
            traces.append(go.Surface(
                x=sx + ox, y=sy + oy, z=sz + oz,
                colorscale=[[0, color], [1, color]],
                opacity=0.13,
                showscale=False,
                hoverinfo="skip",
            ))

            # State vector arrow
            ex, ey, ez = bv[0] + ox, bv[1] + oy, bv[2] + oz
            traces.append(go.Scatter3d(
                x=[ox, ex], y=[oy, ey], z=[oz, ez],
                mode="lines",
                line=dict(color=color, width=7),
                showlegend=False,
            ))
            traces.append(go.Scatter3d(
                x=[ex], y=[ey], z=[ez],
                mode="markers+text",
                marker=dict(size=9, color=color),
                text=[f"{QUBIT_LABELS[i]}<br>S={entropies[i]:.3f}"],
                textposition="top center",
                showlegend=False,
            ))

            # Sphere label at south pole
            traces.append(go.Scatter3d(
                x=[ox], y=[oy], z=[oz - 1.4],
                mode="text",
                text=[QUBIT_LABELS[i]],
                textfont=dict(size=14, color=color),
                showlegend=False,
            ))

        # Entanglement arcs between pairs
        pairs = [(0, 1), (1, 2), (0, 2)]
        pair_entropy = {
            (0, 1): (entropies[0] + entropies[1]) / 2,
            (1, 2): (entropies[1] + entropies[2]) / 2,
            (0, 2): (entropies[0] + entropies[2]) / 2,
        }
        for q0, q1 in pairs:
            ent = pair_entropy[(q0, q1)]
            if ent > 0.05:
                ox0, oy0, oz0 = offsets[q0]
                ox1, oy1, oz1 = offsets[q1]
                # Arc with midpoint lifted
                t = np.linspace(0, 1, 30)
                arc_x = ox0 + t * (ox1 - ox0)
                arc_y = oy0 + t * (oy1 - oy0)
                arc_z = oz0 + t * (oz1 - oz0) + 1.5 * np.sin(math.pi * t) * ent
                traces.append(go.Scatter3d(
                    x=arc_x, y=arc_y, z=arc_z,
                    mode="lines",
                    line=dict(
                        color=ENTANGLEMENT_COLOR,
                        width=max(1, int(ent * 8)),
                        dash="dot",
                    ),
                    name=f"Entanglement {QUBIT_LABELS[q0]}-{QUBIT_LABELS[q1]} S={ent:.3f}",
                    showlegend=True,
                ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=(
                    "ODIN QuantumLogicGate v2.0 — Entanglement View<br>"
                    f"<sup>{repr(register)}</sup>"
                ),
                x=0.5,
            ),
            scene=dict(
                xaxis=dict(range=[-6, 6], title=""),
                yaxis=dict(range=[-2, 2], title=""),
                zaxis=dict(range=[-2, 2], title=""),
                dragmode="orbit",
                aspectmode="manual",
                aspectratio=dict(x=3, y=1, z=1),
            ),
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=0, r=0, b=0, t=80),
            width=1400,
            height=600,
        )
        if show:
            fig.show()
        return fig
