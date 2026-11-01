"""Figures du projet (exploration et résultats), à partir de data/ et resultats/resultats.json.

Usage : python -m src.figures   (écrit resultats/figures/*.png)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
FIGURES = RACINE / "resultats" / "figures"
ACCENT, GRIS, VERT, BLEU = "#b4532a", "#9aa0a6", "#2f7d4f", "#2b6cb0"
COULEURS_MODELES = {"Règle métier": GRIS, "Régression logistique": BLEU, "LightGBM": ACCENT}

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def charger_resultats() -> dict:
    return json.loads((RACINE / "resultats" / "resultats.json").read_text(encoding="utf-8"))


def fig_taux_mensuel(res: dict):
    s = pd.Series(res["taux_impaye_mensuel"])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(s.index, 100 * s.values, color=ACCENT, marker="o", ms=3)
    for mois, nom in (("2024-06", "Tabaski"), ("2025-06", "Tabaski"), ("2026-05", "Tabaski")):
        if mois in s.index:
            ax.annotate(nom, (mois, 100 * s[mois]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_ylabel("Taux d'impayé (%)")
    ax.set_title("Taux d'impayé à 60 jours par mois de facturation")
    ax.set_xticks(s.index[::3])
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def fig_delais(res: dict):
    h = res["histogramme_delais"]
    bords = np.arange(-15, 125, 5)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(bords[:-1], h, width=4.6, align="edge", color=[VERT if b < 0 else ACCENT if b >= 60 else GRIS for b in bords[:-1]])
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(60, color=ACCENT, lw=1, ls="--")
    ax.text(61, max(h) * 0.9, "seuil d'impayé : 60 jours", color=ACCENT, fontsize=8)
    ax.set_xlabel("Délai de paiement après l'échéance (jours, négatif = en avance)")
    ax.set_ylabel("Factures")
    ax.set_title("Distribution des délais de paiement (factures soldées)")
    fig.tight_layout()
    return fig


def fig_par_mode(res: dict):
    s = pd.Series(res["taux_impaye_par_mode"]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(s.index, 100 * s.values, color=ACCENT)
    for i, v in enumerate(s.values):
        ax.text(100 * v + 0.3, i, f"{100 * v:.1f} %", va="center", fontsize=8)
    ax.set_xlabel("Taux d'impayé (%)")
    ax.set_title("Taux d'impayé par mode de paiement habituel")
    fig.tight_layout()
    return fig


def fig_gains(res: dict):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    x = np.linspace(0.05, 1, 20)
    for nom, g in res["gains"].items():
        ax.plot(100 * x, 100 * np.array(g), label=nom, color=COULEURS_MODELES[nom], lw=2 if nom == "LightGBM" else 1.4)
    ax.plot([0, 100], [0, 100], color="black", lw=0.8, ls=":", label="Hasard")
    ax.axvline(20, color=GRIS, lw=0.8, ls="--")
    ax.set_xlabel("Part des factures relancées, des plus risquées aux moins risquées (%)")
    ax.set_ylabel("Part des impayés captés (%)")
    ax.set_title("Courbe de gains (jeu de test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_calibration(res: dict):
    c = pd.DataFrame(res["calibration"])
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, c[["predit", "observe"]].max().max()], [0, c[["predit", "observe"]].max().max()], color="black", lw=0.8, ls=":")
    ax.plot(c["predit"], c["observe"], marker="o", color=ACCENT)
    ax.set_xlabel("Probabilité prédite (moyenne par décile)")
    ax.set_ylabel("Taux d'impayé observé")
    ax.set_title("Calibration de LightGBM")
    fig.tight_layout()
    return fig


def fig_importance(res: dict, nb: int = 12):
    imp = pd.DataFrame(res["importance"]).head(nb).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(imp["libelle"], imp["shap_moyen_abs"], color=ACCENT)
    ax.set_xlabel("Impact moyen sur le score (|SHAP|, log-odds)")
    ax.set_title("Variables les plus influentes (SHAP)")
    fig.tight_layout()
    return fig


def fig_simulation(res: dict):
    s = pd.DataFrame(res["simulation_relance"])
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    couleurs = [GRIS if "hasard" in n else COULEURS_MODELES.get(n, GRIS) for n in s["strategie"]]
    ax.barh(s["strategie"], s["montant_evite_fcfa"] / 1e6, color=couleurs)
    for i, (v, p) in enumerate(zip(s["montant_evite_fcfa"], s["impayes_captes_pct"])):
        ax.text(v / 1e6 + 0.1, i, f"{v / 1e6:.1f} M FCFA · {p:.0f} % des impayés ciblés", va="center", fontsize=8)
    mois = res.get("donnees", {}).get("nb_mois_test")
    ax.set_xlabel(f"Impayés évités{f' sur {mois} mois' if mois else ''} (millions FCFA, hypothèse : 35 % des relances réussissent)")
    ax.set_title("Relancer 20 % des factures : quel ciblage ?")
    ax.set_xlim(0, s["montant_evite_fcfa"].max() / 1e6 * 1.7)
    fig.tight_layout()
    return fig


TOUTES = {"taux_mensuel": fig_taux_mensuel, "delais": fig_delais, "par_mode": fig_par_mode, "gains": fig_gains,
          "calibration": fig_calibration, "importance_shap": fig_importance, "simulation_relance": fig_simulation}


def main() -> None:
    plt.switch_backend("Agg")
    FIGURES.mkdir(parents=True, exist_ok=True)
    res = charger_resultats()
    for nom, f in TOUTES.items():
        fig = f(res)
        fig.savefig(FIGURES / f"{nom}.png", bbox_inches="tight")
        plt.close(fig)
    print(f"{len(TOUTES)} figures -> resultats/figures/")


if __name__ == "__main__":
    main()
