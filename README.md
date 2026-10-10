# HMM-Dekodieren: Map-Matching mit Filter, Glätter und Viterbi – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-hmm-decode-demo.streamlit.app/)** -->

Fünftes Stück der Linie **Zustandsschätzung und Filter** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Es setzt [Kalman-Filter](https://github.com/sebastian-hanisch/kalman-filter-demo) fort und behebt dessen benennbare Schwäche **eine Hypothese, ein Gipfel**: ein Fahrzeug, das auf einer von
mehreren nahen Straßen fährt, ist kein Gauß-Mittelwert dazwischen. Der Zustand ist hier **diskret** (welches Straßenstück), das Modell ein Hidden-Markov-Modell. Aus ihm folgen drei
Auswertungen derselben Messungen: **Filter** (nur Vergangenheit), **Glätter** (Vorwärts-Rückwärts, alle Daten) und **Viterbi** (wahrscheinlichster gemeinsamer Weg, ein kürzester Weg im Trellis).
Im Unterschied zu den Fall-Demos zeigt die Demo ein einziges Verfahren an einem wachsenden Beispiel.

## Warum dieses Problem

Map-Matching ordnet verrauschte GPS-Punkte einem Straßennetz zu. Die Kernfrage: **Was bringt der Blick in die Zukunft, und was garantiert Viterbi tatsächlich?** Naheliegend wäre, dass
Viterbi „die beste Auswertung“ ist; die Messung zeigt etwas anderes.

## Modell

Zustände sind 10-m-Stücke eines 7 × 7-Straßengitters (252, 420 oder 840 Stücke je nach Straßenabstand 30, 50 oder 100 m). Die Messung ist die GPS-Position (Gauß um die Stückmitte).
Der Übergang bevorzugt eine Weiterfahrt um etwa ein Stück je Sekunde entlang des Netzes: Gewicht $\exp(-|d - \ell|/\beta)$ mit $\ell$ = 10 m, $\beta$ = 5 m für Netzabstand $d \le 30$ m
(an die Arbeit von Newson und Krumm angelehnt, aber vereinfacht). Die Übergänge sind dünn besetzt; die Rechnung läuft über Nachbarlisten, die dichte Matrix gibt es nur als Gegenprobe. Die
wahre Fahrt ist ein Zufallsweg, der eine Zelle je Sekunde zurücklegt und nie sofort umkehrt. Formeln im 📐-Expander, Kern in `hmd_algorithm.py`. Kein Straßennetz einer Stadt: ein eigenes Gitter.

## Methodik

Drei Experimente auf Knopfdruck, je über 30 feste Fahrten (`SWEEP_SEEDS`):

1. GPS-Rauschen von 10 bis 60 m: Trefferquote und unmögliche Sprünge der vier Verfahren (nächste Straße, Filter, Glätter, Viterbi).
2. Straßenabstand 30, 50, 100 m: wie viel kostet ein engeres Netz.
3. Kalibrierung: ist die Modell-Sicherheit ehrlich? Mit einer Selbstprüfung auf Daten aus dem Modell selbst.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Mittel über 30 feste Fahrten à 60 s, Trefferquote = Anteil der Sekunden mit Netzabstand zur wahren Stelle höchstens 15 m; `tests/test_claims.py` rechnet jede nach.

| Frage | Befund |
|---|---|
| Was bringt das Modell gegen „nächste Straße“? | Bei 20 m Rauschen (Netz 100 m): nächste Straße **39 %**, Filter **62 %**, Glätter **90 %**, Viterbi **87 %**. |
| Woher kommt der Gewinn? | Vom **Blick in die Zukunft**: Glätter und Viterbi liegen bei allen fünf Rauschstufen weniger als 7 Punkte auseinander, der Glätter ist bei 10 bis 30 m etwas besser (98/90/79 % gegen 96/87/75 %). |
| Was garantiert Viterbi dann? | Einen **zulässigen** Weg: **0 %** unmögliche Sprünge in allen Einstellungen. Der Filter springt bei 30 m in etwa **4 %** der Sekunden an unerreichbare Stellen, bei 60 m in etwa **10 %**, die „nächste Straße“ in 11 % bis über 80 %. Der Glätter liegt unter 1 %, garantiert es aber nicht. |
| Engeres Netz | Bei 30 m Rauschen: Glätter **56 %** (Straßenabstand 30 m) gegen **79 %** (100 m); Viterbi **48 %** gegen **75 %**. |
| Sind die Wahrscheinlichkeiten ehrlich? | Auf Daten aus dem Modell selbst ja: Filter 54 % Modell-Sicherheit gegen 56 % Treffer, Glätter 68 % gegen 67 %. Auf der simulierten Fahrt (30 m, Netz 50 m) ist der **Filter überselbstsicher** (52 % gegen 41 %), der Glätter nicht (68 % gegen 72 %); glaubt das Modell dem GPS nur halb so viel Rauschen, sind es 72 % gegen 42 %. |

## Ehrliche Grenzen

- **Die Wahrheit ist ein Zufallsweg auf einem Gitter, kein echtes Straßennetz.** Es gibt keine Einbahnstraßen, Geschwindigkeiten, Brücken oder echte GPS-Fehlermuster.
- **Die Übergangsparameter sind von Hand gesetzt** ($\beta$ = 5 m) und nicht aus Daten gelernt; die Überselbstsicherheit des Filters zeigt, was passiert, wenn das Modell nicht zur Fahrt passt.
- **Treffer sind an einem Radius gemessen** (15 m); ein Treffer heißt nicht „richtige Straße“, wenn parallele Straßen näher als 15 m liegen (hier ab 30 m Abstand nicht der Fall).
- **Viterbi ist hier nicht besser als der Glätter.** Das ist die Aussage der Messung, nicht eine Frage der Implementierung: die Rechnung ist mit Aufzählung aller Pfade und mit Dijkstra im Trellis geprüft.
- **Kleine Stichprobe:** 30 Fahrten; einzelne Prozentpunkte sind Zufall.
- **Die Modell-Sicherheit** ist als Wahrscheinlichkeitsmasse im Trefferradius um die wahrscheinlichste Stelle definiert; sie ist nicht die Wahrscheinlichkeit, auf der „richtigen Straße“ zu sein.

## Verifikation

- **Kern:** Netzgröße, Nachbarschaften, Übergangsgewichte, Messmodell, Rauschfreiheit (alle drei Verfahren finden den wahren Weg) gegen Handrechnung auf dem Mini-Netz (`tests/test_algorithm.py`).
- **Orakel** (`tests/test_oracle_hmm_decode.py`, anderer Rechenweg):
  - **Aufzählung aller Pfade** (Mini-Netz): Viterbi-Pfad, Glätter-Randverteilungen und Filterverteilungen aller Zeitpunkte.
  - **Viterbi = kürzester Weg im Trellis** (`networkx`, Dijkstra mit Kosten −log A − log b).
  - **Dünn besetzte gegen eigenständige dichte Rechnung** auf dem großen Netz (Filter, Glätter, Pfadwert).
  - **Lokale Optimalität** des Viterbi-Pfads und besser als der Glätter-Pfad bezüglich der Pfadwahrscheinlichkeit.
  - **Selbstkalibrierung** auf Daten aus dem Modell.
  - Angezeigte Kennzahlen gegen unabhängige Neuberechnung.
- **Zahlen im README:** `tests/test_claims.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, Regler an Minimum und Maximum, jede Auswahloption, Experiment-Knöpfe) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `hmd_algorithm.py` | Kern: Straßengitter, dünne Übergänge, Messmodell, Filter, Glätter, Viterbi |
| `hmd_scenario.py` | Zufallsfahrt und GPS-Messungen (ein Generator, Regler skalieren denselben Zufall) |
| `hmd_evaluation.py` | Kennzahlen, Sweeps, Kalibrierung, Selbstprüfung |
| `hmd_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `hmd_presets.py` | Permalink (Begrenzen/Einrasten), Presets, Zufalls-Seed-Button |
| `hmd_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `hmd_pdf_export.py` | PDF-Auswertung (fpdf2) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit bzw. Laufzeit plus pytest/networkx |
| `tests/` | Einheiten, Orakel, Behauptungen, Presets, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Parameter lernen** (Baum-Welch) und **Kandidatenauswahl** für große Netze: nicht umgesetzt.
- **Echte Straßendaten, Einbahnstraßen, Richtung und Geschwindigkeit im Zustand:** nicht eingebunden.
- **Festverzögerungs-Glätter** für den HMM-Fall: nur über den Vergleich Filter/Glätter angedeutet.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
