"""HMM-Dekodieren: Map-Matching mit Filter, Glätter und Viterbi - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Fahrzeug fährt durch ein Straßengitter, GPS misst seine Position verrauscht. Der Zustand ist jetzt DISKRET: welches Straßenstück.
Ein Hidden-Markov-Modell liefert drei Auswertungen derselben Messungen: Filter (nur Vergangenheit), Glätter (alle Daten) und Viterbi
(wahrscheinlichster gemeinsamer Pfad).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import hmd_constants as C
import hmd_evaluation as ev
from hmd_evaluation import Settings, analyse
from hmd_pdf_export import build_pdf
from hmd_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from hmd_visualization import build_calibration_chart, build_posterior_map, build_sigma_hits, build_sigma_jumps, build_spacing_chart, build_track_map

st.set_page_config(page_title="HMM-Dekodieren – Sebastian Hanisch", layout="wide")


def de(x, nd=1):
    """Dezimalkomma statt Punkt."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f} %".replace(".", ",")


def factor_label(f):
    return "×" + f"{f:g}".replace(".", ",")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep_sigma(settings):
    return ev.sweep_sigma(settings)


@st.cache_data(show_spinner=False)
def _sweep_spacing(settings):
    return ev.sweep_spacing(settings)


@st.cache_data(show_spinner=False)
def _sweep_calibration(settings):
    return {"rows": ev.sweep_calibration(settings), "self": ev.self_calibration(settings)}


def run_experiment(key, fn, settings, label):
    """Teure Sweeps nur auf Knopfdruck: das Ergebnis bleibt sichtbar, solange die Einstellungen gleich bleiben."""
    if st.button(label, key=f"run_{key}"):
        st.session_state[f"done_{key}"] = settings
    if st.session_state.get(f"done_{key}") != settings:
        st.info("Noch nicht gerechnet für diese Einstellungen. Der Knopf startet die Rechnung (einige Sekunden); danach bleibt das Ergebnis stehen.")
        return None
    with st.spinner("Rechne … (je nach Rechner einige Sekunden)"):
        return fn(settings)


st.title("🗺️ HMM-Dekodieren: Map-Matching mit Filter, Glätter und Viterbi")
st.markdown(
    """
Ein Gauß-Filter kennt **eine** Hypothese: einen Mittelwert mit einer Unsicherheit. Fährt das Fahrzeug aber auf **einer von mehreren nahen Straßen**, sind das mehrere getrennte
Möglichkeiten, und ein Mittelwert dazwischen liegt auf keiner. Hier ist der Zustand deshalb **diskret**: welches Straßenstück (10 m lang). Ein **Hidden-Markov-Modell (HMM)**
verbindet dafür zwei Dinge: Wie wahrscheinlich ist eine GPS-Messung, wenn das Fahrzeug auf diesem Stück ist? (**Messmodell**) Und: Auf welche Stücke fährt es von diesem aus weiter? (**Übergangsmodell**)

Daraus gibt es **drei Auswertungen derselben Messungen**: den **Filter** (jede Sekunde nur mit der Vergangenheit, wie im Kalman-Filter), den **Glätter** (jede Sekunde mit allen Daten)
und **Viterbi** (der wahrscheinlichste *gesamte* Weg). Die Demo ist das fünfte Stück der Linie **Zustandsschätzung und Filter**. Im Unterschied zu den Fall-Demos zeigt sie ein einziges
Verfahren an einem wachsenden Beispiel.

Weiter unten: drei Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktionieren die drei Auswertungen", expanded=True):
    st.markdown(
        """
Das Modell kennt für jede Sekunde eine Wahrscheinlichkeit je Straßenstück. Die drei Auswertungen unterscheiden sich darin, **welche Messungen** sie dafür nutzen und **was** sie ausgeben:

1. **Filter:** Wahrscheinlichkeit je Stück gegeben alle Messungen **bis jetzt**; ausgegeben wird je Sekunde das wahrscheinlichste Stück. Echtzeitfähig, aber die Folge der Stücke muss
   nicht zusammenhängen.
2. **Glätter (Vorwärts-Rückwärts):** Wahrscheinlichkeit je Stück gegeben **alle** Messungen, auch die späteren. Ausgegeben wird je Sekunde das wahrscheinlichste Stück.
3. **Viterbi:** nicht je Sekunde, sondern der **eine** Weg mit der größten Gesamtwahrscheinlichkeit. Er ist ein **kürzester Weg im Trellis** (Zeit × Straßenstücke, Kantenkosten = −log der Wahrscheinlichkeiten)
   und damit per Konstruktion ein **zusammenhängender** Weg.

**Merke:** Der Sprung vom Filter zu den beiden anderen kommt vor allem vom **Blick in die Zukunft**. Viterbi garantiert zusätzlich, dass der Weg ein möglicher Weg ist.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Wirklichkeit**")
    sigma = st.slider("GPS-Rauschen (m)", *bounds("sigma_slider"), key="sigma_slider", step=1.0,
                      help="Streuung der GPS-Position um die wahre Position (Standardabweichung je Achse, in Metern).")
    spacing = st.select_slider("Straßenabstand (m)", C.SPACING_CHOICES, key="spacing_select",
                               help="Abstand zwischen parallelen Straßen. Je enger, desto leichter verwechselt man sie.")
    st.markdown("**Was das Modell glaubt**")
    assumed = st.select_slider("Rauschen laut Modell", C.ASSUMED_CHOICES, key="assumed_select", format_func=factor_label,
                               help="Verhältnis des vom Modell angenommenen zum wahren GPS-Rauschen. ×1 heißt: das Modell kennt die Wahrheit.")
    seed = st.number_input("Zufalls-Seed der Fahrt", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Fahrt generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"sigma_slider": float(sigma), "spacing_select": int(spacing), "assumed_select": float(assumed), "seed_input": int(seed)})

settings = Settings(float(sigma), int(spacing), float(assumed), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m = st.columns(4)
for col, k in zip(m, C.METHODS):
    delta = "Referenz" if k == "snap" else f"{100 * (a.hit[k] - a.hit['snap']):+.0f} Punkte gegen nächste Straße"
    col.metric(C.METHOD_LABELS[k], pct(a.hit[k]), delta=delta, delta_color="off" if k == "snap" else "normal",
               help=f"Trefferquote: Anteil der Sekunden, in denen die Schätzung höchstens {C.HIT_RADIUS:.0f} m (Netzabstand) von der wahren Position liegt.")

st.markdown("**Fahrt auf dem Straßennetz** – Wahl der Wege, die gezeigt werden:")
show_labels = {"truth": "Wahrheit", **{k: C.METHOD_LABELS[k] for k in C.METHODS}}
chosen = st.multiselect("Wege", list(show_labels), default=["truth", "filter", "viterbi"], format_func=lambda k: show_labels[k], key="show_paths", label_visibility="collapsed")
st.plotly_chart(build_track_map(a, set(chosen)), width="stretch", key="track_map")

st.markdown("**Unmögliche Sprünge** (aufeinanderfolgende Schätzungen mehr als 30 m auseinander im Netz) und **Modell-Sicherheit**:")
st.markdown(
    "| Auswertung | Trefferquote | Unmögliche Sprünge | Modell-Sicherheit (Masse im Trefferradius) |\n|---|---|---|---|\n"
    + "\n".join(
        f"| {C.METHOD_LABELS[k]} | {pct(a.hit[k])} | {pct(a.jumps[k], 1)} | "
        + (pct(a.conf_filter) if k == "filter" else pct(a.conf_smooth) if k == "smoother" else "–") + " |"
        for k in C.METHODS
    )
)

st.markdown("**Was das Modell zu einer Sekunde glaubt** – Wahrscheinlichkeit je Straßenstück, links nur mit der Vergangenheit, rechts mit allen Messungen:")
t_show = st.slider("Zeitpunkt (s)", 0, C.T_STEPS - 1, C.T_STEPS // 2, key="t_show", help="Welche Sekunde der Fahrt gezeigt wird.")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Filter**")
    st.plotly_chart(build_posterior_map(a, "filter", int(t_show)), width="stretch", key="post_filter_map")
with c2:
    st.markdown("**Glätter**")
    st.plotly_chart(build_posterior_map(a, "smoother", int(t_show)), width="stretch", key="post_smooth_map")

st.download_button("📄 Auswertung als PDF", data=build_pdf(a), file_name="hmm_dekodieren_auswertung.pdf", mime="application/pdf", key="pdf_button")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was bringt der Blick in die Zukunft, und was garantiert Viterbi?")
gain = a.hit["smoother"] - a.hit["filter"]
diff = a.hit["viterbi"] - a.hit["smoother"]
if abs(diff) <= 0.03:
    vs = "Viterbi und der Glätter liegen auf dieser Fahrt dicht beieinander: der Gewinn kommt vom Blick in die Zukunft."
elif diff > 0:
    vs = (f"Viterbi ist auf **dieser** Fahrt {100 * diff:.0f} Punkte besser als der Glätter; im Mittel über {len(C.SWEEP_SEEDS)} Fahrten liegen beide dicht beieinander "
          "(Experiment 1): ein einzelner Weg kann so ausgehen.")
else:
    vs = f"Der Glätter ist auf dieser Fahrt {-100 * diff:.0f} Punkte besser als Viterbi: die wahrscheinlichste Einzel-Sekunde ist nicht der wahrscheinlichste gemeinsame Weg."
if a.jumps["filter"] > 0.005:
    st.info(
        f"Der Filter erreicht {pct(a.hit['filter'])}, der Glätter {pct(a.hit['smoother'])} (**{100 * gain:+.0f} Punkte**) und Viterbi {pct(a.hit['viterbi'])}. "
        f"Der Filter springt in {pct(a.jumps['filter'], 1)} der Sekunden an eine Stelle, die vom Vorgänger aus gar nicht erreichbar ist; Viterbi in {pct(a.jumps['viterbi'], 1)}, "
        f"der Glätter in {pct(a.jumps['smoother'], 1)}. {vs}"
    )
else:
    st.success(
        f"Bei diesem Rauschen ist die Lage eindeutig: Filter {pct(a.hit['filter'])}, Glätter {pct(a.hit['smoother'])}, Viterbi {pct(a.hit['viterbi'])}; unmögliche Sprünge "
        f"praktisch keine ({pct(a.jumps['filter'], 1)} beim Filter). {vs}"
    )
st.caption("Eine einzelne Fahrt zeigt nur, was hier passiert ist; die Experimente unten mitteln über 30 feste Fahrten.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie hängt alles vom GPS-Rauschen ab?")
st.caption(f"Mittel über {len(C.SWEEP_SEEDS)} feste Fahrten; Straßenabstand und Modellannahme wie in der Seitenleiste, das GPS-Rauschen läuft von 10 bis 60 m.")
rows = run_experiment("sigma", _sweep_sigma, settings, f"▶ Experiment 1 rechnen ({len(C.SIGMA_SWEEP) * len(C.SWEEP_SEEDS)} Fahrten)")
if rows is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Trefferquote**")
        st.plotly_chart(build_sigma_hits(rows), width="stretch", key="sigma_hits_chart")
    with c2:
        st.markdown("**Unmögliche Sprünge**")
        st.plotly_chart(build_sigma_jumps(rows), width="stretch", key="sigma_jumps_chart")
    mid = next(r for r in rows if r["sigma"] == 30.0)
    top = rows[-1]
    st.info(
        f"Bei 30 m Rauschen: nächste Straße {pct(mid['hit']['snap'])}, Filter {pct(mid['hit']['filter'])}, Glätter {pct(mid['hit']['smoother'])}, Viterbi {pct(mid['hit']['viterbi'])}. "
        f"Der Filter springt dann in {pct(mid['jumps']['filter'], 1)} der Sekunden an unerreichbare Stellen, Viterbi nie ({pct(mid['jumps']['viterbi'], 1)}). Bei 60 m: Glätter {pct(top['hit']['smoother'])}, Viterbi {pct(top['hit']['viterbi'])}. "
        "**Glätter und Viterbi liegen bei der Trefferquote dicht beieinander; der Glätter ist an vielen Stellen sogar etwas besser.** Viterbi garantiert dafür einen **zulässigen** Weg."
    )

st.subheader("🔬 Experiment 2: Wie wirkt ein engeres Straßennetz?")
st.caption(f"Mittel über {len(C.SWEEP_SEEDS)} feste Fahrten bei dem GPS-Rauschen der Seitenleiste; die Modellannahme wie dort, der Straßenabstand läuft über 30, 50 und 100 m.")
srows = run_experiment("spacing", _sweep_spacing, settings, f"▶ Experiment 2 rechnen ({len(C.SPACING_CHOICES) * len(C.SWEEP_SEEDS)} Fahrten)")
if srows is not None:
    st.plotly_chart(build_spacing_chart(srows), width="stretch", key="spacing_chart")
    tight, wide = srows[0], srows[-1]
    st.info(
        f"Bei {tight['spacing']} m Straßenabstand erreicht der Glätter {pct(tight['hit']['smoother'])}, bei {wide['spacing']} m {pct(wide['hit']['smoother'])}: "
        f"parallele Straßen liegen näher als das GPS-Rauschen, die Verwechslung nimmt zu. Der Filter fällt von {pct(wide['hit']['filter'])} auf {pct(tight['hit']['filter'])}."
    )

st.subheader("🔬 Experiment 3: Sind die Wahrscheinlichkeiten ehrlich?")
st.caption(
    f"Das Modell sagt zu jeder Sekunde, wie sicher es ist, dass die Schätzung höchstens {C.HIT_RADIUS:.0f} m daneben liegt (Wahrscheinlichkeitsmasse um die wahrscheinlichste Stelle). "
    f"Dagegen die tatsächliche Trefferquote; {len(C.SWEEP_SEEDS)} feste Fahrten, GPS-Rauschen und Straßenabstand wie in der Seitenleiste, das Modell glaubt dem GPS nur halb so viel, genau so viel oder doppelt so viel Rauschen zu."
)
cal = run_experiment("calibration", _sweep_calibration, settings, f"▶ Experiment 3 rechnen ({len(C.ASSUMED_CHOICES) * len(C.SWEEP_SEEDS)} Fahrten plus Selbstprüfung)")
if cal is not None:
    st.plotly_chart(build_calibration_chart(cal["rows"]), width="stretch", key="calibration_chart")
    one = next(r for r in cal["rows"] if r["assumed"] == 1.0)
    half = next(r for r in cal["rows"] if r["assumed"] == 0.5)
    sc = cal["self"]
    st.info(
        f"**Selbstprüfung:** Werden die Fahrten aus dem Übergangsmodell selbst gezogen, stimmen Sicherheit und Trefferquote überein (Filter {pct(sc['filter']['conf'])} gegen {pct(sc['filter']['hit'])}, "
        f"Glätter {pct(sc['smoother']['conf'])} gegen {pct(sc['smoother']['hit'])}): die Rechnung ist in Ordnung. Auf der simulierten Fahrt mit genauem Rauschen liegt der Filter bei "
        f"{pct(one['filter']['conf'])} Sicherheit gegen {pct(one['filter']['hit'])} Trefferquote, der Glätter bei {pct(one['smoother']['conf'])} gegen {pct(one['smoother']['hit'])}; "
        f"glaubt das Modell dem GPS zu sehr (×0,5), sind es {pct(half['filter']['conf'])} gegen {pct(half['filter']['hit'])} beim Filter. Die Abweichung kommt von der Modellannahme "
        "(Übergänge von Hand gesetzt, die wahre Fahrt hält sich nicht daran), nicht von der Rechnung."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Übergangswahrscheinlichkeiten stimmen** | Sie sind hier von Hand gesetzt (Weiterfahrt um etwa ein Stück je Sekunde, Streuung 5 m). Passen sie nicht zur Fahrt, sind die Wahrscheinlichkeiten nicht ehrlich (Experiment 3). | Parameter aus Daten lernen (Baum-Welch) |
| **Das Rauschen ist bekannt und gaußsch** | Systematische Fehler (Hochhausschluchten) machen die Messung auf einer falschen Straße plausibel. | Robuste Messmodelle |
| **Der Zustandsraum ist klein genug** | Hier 252 bis 840 Stücke; für ein Stadtnetz braucht man eine Kandidatenauswahl je Messung, wie sie in der Praxis üblich ist. | Kandidatenauswahl |
| **Ein Fahrzeug, eine Spur** | Mehrere Fahrzeuge brauchen eine Zuordnung der Messungen. | Mehrzielverfolgung |
| **Alles auf einmal verfügbar (Glätter, Viterbi)** | In Echtzeit gibt es nur den Filter oder einen Glätter mit Verzögerung. | Festverzögerungs-Glätter |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Zustände $x_t \in \{1, \dots, N\}$ (Straßenstücke), Start $\pi$, Übergang $A_{ij} = P(x_{t+1}=j \mid x_t=i)$, Messmodell $b_j(z) = p(z \mid x=j)$.
Hier: $b_j(z) \propto \exp(-\lVert z - c_j\rVert^2 / 2\sigma^2)$ (Zellenmitte $c_j$) und
$A_{ij} \propto \exp(-|d_{ij} - \ell|/\beta)$ für Netzabstand $d_{ij} \le 3\ell$ ($\ell = 10$ m Zellenlänge, $\beta = 5$ m).

**Filter (Vorwärts).** $\alpha_t(j) \propto b_j(z_t)\sum_i \alpha_{t-1}(i) A_{ij}$, normiert; $\alpha_t(j) = P(x_t = j \mid z_{1:t})$. Je Sekunde $\hat x_t = \arg\max_j \alpha_t(j)$.

**Glätter (Vorwärts-Rückwärts).** $\beta_t(i) \propto \sum_j A_{ij}\, b_j(z_{t+1})\,\beta_{t+1}(j)$, $\gamma_t(j) \propto \alpha_t(j)\beta_t(j) = P(x_t = j \mid z_{1:T})$. Je Sekunde $\hat x_t = \arg\max_j \gamma_t(j)$.
Die Folge dieser Maxima ist **nicht** der wahrscheinlichste Weg und muss nicht zusammenhängen.

**Viterbi.** $\delta_t(j) = \max_i\bigl[\delta_{t-1}(i) + \log A_{ij}\bigr] + \log b_j(z_t)$, Rückverfolgung über die Argmax-Zeiger; der Weg maximiert $P(x_{1:T}, z_{1:T})$.
Mit Kosten $c = -\log A_{ij} - \log b_j(z_t)$ ist das ein **kürzester Weg** im Trellis (Zeitschicht $t$, Knoten $j$): dasselbe Problem wie in der Kürzeste-Wege-Linie, nur auf einem Graphen aus $T \cdot N$ Knoten.

**Aufwand.** Mit den dünnen Übergängen $O(T \cdot N \cdot K)$ je Verfahren ($K$ = Nachbarn je Stück, hier höchstens 15), dicht $O(T N^2)$.

**Literatur.** P. Newson, J. Krumm (2009), *Hidden Markov Map Matching Through Noise and Sparseness*, Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic
Information Systems (Seattle). Dort sind Zustände Straßensegmente, die Messwahrscheinlichkeit ein Gauß um den Abstand zum Segment und der Übergang exponentiell in der Differenz von Luftlinie
und Routenlänge; die Demo verwendet eine vereinfachte, an diese Arbeit angelehnte Form auf einem Gitter.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
