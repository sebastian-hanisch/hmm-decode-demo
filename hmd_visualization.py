"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import hmd_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=70), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _roads(fig, net):
    fig.add_trace(go.Scatter(x=net.coords[:, 0], y=net.coords[:, 1], mode="markers", marker=dict(color=C.COLOR_ROAD, size=3), name="Straßennetz",
                             hoverinfo="skip"))


def _map_layout(fig, net, height=470):
    """Karte mit gleichem Maßstab: scaleanchor nur mit autorange; den Rand erzwingen zwei unsichtbare Marker an den Ecken (DEMO-PLAYBOOK §3)."""
    lim = float(net.coords.max()) + 15
    fig.add_trace(go.Scatter(x=[-15, lim], y=[-15, lim], mode="markers", marker=dict(opacity=0, size=1), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title_text="x (m)")
    fig.update_yaxes(title_text="y (m)", scaleanchor="x", scaleratio=1)
    return _base(fig, height)


def build_track_map(a, show):
    """Straßennetz, GPS-Messungen und die gewählten Wege (Wahrheit, Schätzungen)."""
    net, tr = a.net, a.track
    fig = go.Figure()
    _roads(fig, net)
    fig.add_trace(go.Scatter(x=tr.z[:, 0], y=tr.z[:, 1], mode="markers", marker=dict(color=C.COLOR_RAW, size=5), name="GPS-Messung",
                             hovertemplate="t = %{pointNumber} s<extra>GPS</extra>"))
    if "truth" in show:
        fig.add_trace(go.Scatter(x=tr.pos[:, 0], y=tr.pos[:, 1], mode="lines", line=dict(color=C.COLOR_TRUTH, width=5), name="Wahrheit"))
    for m in C.METHODS:
        if m in show:
            p = net.coords[a.paths[m]]
            fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines+markers", line=dict(color=C.COLORS[m], width=2, dash="dot" if m == "snap" else "solid"),
                                     marker=dict(size=4), name=C.METHOD_LABELS[m]))
    return _map_layout(fig, net)


def build_posterior_map(a, which, t):
    """Wahrscheinlichkeit jeder Zelle zum Zeitpunkt t (Filter: nur Daten bis t, Glätter: alle Daten), samt Wahrheit und Messung."""
    net, tr = a.net, a.track
    post = (a.post_filter if which == "filter" else a.post_smooth)[t]
    fig = go.Figure()
    order = np.argsort(post)
    fig.add_trace(go.Scatter(x=net.coords[order, 0], y=net.coords[order, 1], mode="markers",
                             marker=dict(color=post[order], colorscale="Viridis", size=4 + 22 * np.sqrt(post[order] / max(post.max(), 1e-12)),
                                         showscale=True, colorbar=dict(title="Wahrscheinlichkeit", thickness=12)),
                             name="Wahrscheinlichkeit je Zelle", hovertemplate="p = %{marker.color:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[tr.pos[t, 0]], y=[tr.pos[t, 1]], mode="markers", marker=dict(color=C.COLOR_TRUTH, size=14, symbol="diamond",
                                                                                    line=dict(color="white", width=2)), name="wahre Position"))
    fig.add_trace(go.Scatter(x=[tr.z[t, 0]], y=[tr.z[t, 1]], mode="markers", marker=dict(color="#333", size=12, symbol="x"), name="GPS-Messung"))
    return _map_layout(fig, net, 430)


def _method_lines(fig, xs, rows, field, labels=None):
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=xs, y=[r[field][m] for r in rows], mode="lines+markers", name=C.METHOD_LABELS[m],
                                 line=dict(color=C.COLORS[m], width=3 if m in ("smoother", "viterbi") else 2, dash="dot" if m == "snap" else "solid")))


def build_sigma_hits(rows):
    fig = go.Figure()
    _method_lines(fig, [r["sigma"] for r in rows], rows, "hit")
    fig.update_xaxes(title_text="GPS-Rauschen (m)")
    fig.update_yaxes(title_text="Trefferquote (Anteil der Sekunden)", range=[0, 1.02], tickformat=".0%")
    return _base(fig)


def build_sigma_jumps(rows):
    fig = go.Figure()
    _method_lines(fig, [r["sigma"] for r in rows], rows, "jumps")
    fig.update_xaxes(title_text="GPS-Rauschen (m)")
    fig.update_yaxes(title_text="Unmögliche Sprünge (Anteil)", range=[0, 1.0], tickformat=".0%")
    return _base(fig)


def build_spacing_chart(rows):
    labels = [f"{r['spacing']} m" for r in rows]
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Bar(x=labels, y=[r["hit"][m] for r in rows], name=C.METHOD_LABELS[m], marker_color=C.COLORS[m]))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Straßenabstand", type="category")
    fig.update_yaxes(title_text="Trefferquote", range=[0, 1.02], tickformat=".0%")
    return _base(fig)


def build_calibration_chart(rows):
    labels = [f"×{r['assumed']:g}".replace(".", ",") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["filter"]["conf"] for r in rows], name="Filter: Sicherheit laut Modell", marker_color="rgba(201,123,42,0.45)"))
    fig.add_trace(go.Bar(x=labels, y=[r["filter"]["hit"] for r in rows], name="Filter: tatsächliche Trefferquote", marker_color="#c97b2a"))
    fig.add_trace(go.Bar(x=labels, y=[r["smoother"]["conf"] for r in rows], name="Glätter: Sicherheit laut Modell", marker_color="rgba(76,120,168,0.45)"))
    fig.add_trace(go.Bar(x=labels, y=[r["smoother"]["hit"] for r in rows], name="Glätter: tatsächliche Trefferquote", marker_color="#4c78a8"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Rauschen, das das Modell annimmt (Vielfaches des wahren)", type="category")
    fig.update_yaxes(title_text="Anteil", range=[0, 1.02], tickformat=".0%")
    return _base(fig)
