"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt und rastet ein."""

import hmd_constants as C
import hmd_evaluation as ev
import hmd_presets as P


def _settings_from(p):
    return ev.Settings(sigma=p["sg"], spacing=p["sp"], assumed=p["as"], seed=p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, p in C.PRESETS.items():
        assert C.SIGMA_MIN <= p["sg"] <= C.SIGMA_MAX, name
        assert p["sp"] in C.SPACING_CHOICES and p["as"] in C.ASSUMED_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert _settings_from(C.PRESETS["Standard (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_clamps_numbers_and_snaps_choices():
    sg = P.SETTING_SPECS["sigma_slider"]
    assert P.clamp_value(sg, 1e6) == C.SIGMA_MAX and P.clamp_value(sg, -1.0) == C.SIGMA_MIN
    sp = P.SETTING_SPECS["spacing_select"]
    assert P.clamp_value(sp, 50) == 50 and P.clamp_value(sp, 75) is None


def test_the_presets_show_what_their_help_text_claims():
    base = C.PRESETS["Standard (Voreinstellung)"]
    big = C.PRESETS["Rauschen größer als der Straßenabstand"]
    assert big["sg"] > big["sp"] and big["sp"] == min(C.SPACING_CHOICES)
    assert C.PRESETS["Dichtes Netz, mäßiges Rauschen"]["sp"] < base["sp"]
    assert C.PRESETS["Modell glaubt dem GPS zu sehr"]["as"] < 1.0
