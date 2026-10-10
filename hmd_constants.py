"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# Straßengitter: NODES x NODES Kreuzungen, Straßen in SPACING-Abstand, Zustände = Straßenstücke („Zellen“) von CELL Metern
NODES = 7
CELL = 10.0                       # Länge einer Zelle in m = Weg je Sekunde bei 10 m/s
SPACING_CHOICES = (30, 50, 100)   # Straßenabstand in m (Vielfache von CELL)
DEFAULT_SPACING = 100
T_STEPS = 60                      # Sekunden je Fahrt
N_BURN = 0                        # alle Schritte zählen
HIT_RADIUS = 15.0                 # „getroffen“: Netzabstand zur wahren Zelle höchstens so groß (m)
JUMP_DIST = 30.0                  # „unmöglicher Sprung“: Netzabstand aufeinanderfolgender Schätzungen darüber (m)
MAX_HOPS = 4                      # Netzabstände werden bis zu so vielen Zellen gerechnet, größere gelten als „fern“
BETA = 5.0                        # Übergangsmodell: Wahrscheinlichkeit ~ exp(-|Schrittlänge - CELL| / BETA)

# Reglergrenzen
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 5.0, 60.0, 20.0
ASSUMED_CHOICES = (0.5, 1.0, 2.0)
DEFAULT_ASSUMED = 1.0
SEED_MAX, DEFAULT_SEED = 999999, 35

# feste Instanzen für alle Sweeps: nie ein einzelner Seed als Befund
SWEEP_SEEDS = tuple(range(100000, 100030))
SIGMA_SWEEP = (10.0, 20.0, 30.0, 45.0, 60.0)

METHODS = ("snap", "filter", "smoother", "viterbi")
METHOD_LABELS = {"snap": "Nächste Straße (ohne Modell)", "filter": "HMM-Filter (online)", "smoother": "HMM-Glätter (alle Daten)", "viterbi": "Viterbi (bester Pfad)"}

COLOR_TRUTH = "#2F6B65"
COLOR_RAW = "rgba(150,150,150,0.8)"
COLOR_ROAD = "rgba(120,120,120,0.35)"
COLORS = {"snap": "#7a7a7a", "filter": "#c97b2a", "smoother": "#4c78a8", "viterbi": "#7a5fa6"}

PRESETS = {
    "Standard (Voreinstellung)": {"sg": DEFAULT_SIGMA, "sp": DEFAULT_SPACING, "as": DEFAULT_ASSUMED, "seed": DEFAULT_SEED},
    "Rauschen größer als der Straßenabstand": {"sg": 45.0, "sp": 30, "as": 1.0, "seed": DEFAULT_SEED},
    "Dichtes Netz, mäßiges Rauschen": {"sg": 20.0, "sp": 30, "as": 1.0, "seed": DEFAULT_SEED},
    "Modell glaubt dem GPS zu sehr": {"sg": 30.0, "sp": 50, "as": 0.5, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard (Voreinstellung)": "20 m GPS-Rauschen auf einem Netz mit 100 m Straßenabstand: meist eindeutig, wo das Fahrzeug fährt.",
    "Rauschen größer als der Straßenabstand": "45 m Rauschen bei 30 m Straßenabstand: das GPS kann benachbarte Straßen nicht mehr unterscheiden, nur der Zusammenhang des Wegs hilft.",
    "Dichtes Netz, mäßiges Rauschen": "30 m Straßenabstand, 20 m Rauschen: viele plausible Nachbarstraßen.",
    "Modell glaubt dem GPS zu sehr": "Das Modell nimmt halb so viel Rauschen an, wie es wirklich gibt: es wird überselbstsicher.",
}
