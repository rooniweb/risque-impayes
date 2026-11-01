"""Construit et exécute le notebook d'analyse (notebooks/analyse_risque_impayes.ipynb).

Le notebook raconte l'étude pas à pas en appelant les fonctions de src/ : une seule source de vérité.
Usage : python -m src.construire_notebook
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "notebooks" / "analyse_risque_impayes.ipynb"

CELLULES = [
    ("md", """# Prédire le risque d'impayé d'une facture télécom

**Question métier** : dès l'émission d'une facture, peut-on estimer la probabilité qu'elle ne soit pas réglée
60 jours après son échéance, pour concentrer les relances là où elles servent ?

- Données **entièrement fictives**, simulées pour un opérateur télécom imaginaire d'Afrique de l'Ouest
  (`src/generer_donnees.py`, graine fixe : tout est reproductible).
- Unité d'analyse : **la facture** (≈ 100 000 factures mensuelles de 5 000 comptes postpayés, janvier 2024 à septembre 2026).
- Cible : `risque_impaye` = 1 si la facture n'est pas soldée 60 jours après l'échéance."""),
    ("code", """import sys, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import preparer_dataset as prep, entrainer as ent, figures as fig
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
comptes, factures = prep.charger_brut()
print(comptes.shape, factures.shape)
factures.head()"""),
    ("md", """## 1. Exploration
Le délai de paiement est la base de tout : `date_solde − date_echeance`. Les factures récentes (moins de 60 jours
après l'échéance au 31/10/2026) n'ont pas encore de cible : ce sont elles que le modèle devra noter."""),
    ("code", """connues = factures[factures["risque_impaye"].notna()]
print(f"Factures à cible connue : {len(connues):,} / {len(factures):,}")
print(f"Taux d'impayé à 60 jours : {connues['risque_impaye'].astype(int).mean():.1%}")
factures["delai_paiement_jours"].describe()"""),
    ("code", """exploration = {
    "taux_impaye_mensuel": connues.assign(y=connues["risque_impaye"].astype(int)).groupby("periode")["y"].mean().to_dict(),
    "histogramme_delais": np.histogram(factures["delai_paiement_jours"].dropna().astype(float).clip(-15, 120),
                                       bins=np.arange(-15, 125, 5))[0].tolist(),
    "taux_impaye_par_mode": connues.merge(comptes, on="id_compte").assign(y=lambda d: d["risque_impaye"].astype(int))
                                   .groupby("mode_paiement")["y"].mean().to_dict(),
}
fig.fig_taux_mensuel(exploration); fig.fig_delais(exploration); fig.fig_par_mode(exploration); plt.show()"""),
    ("md", """**Lecture** : pics d'impayés les mois de Tabaski (dépenses familiales) et à la rentrée ; le prélèvement
automatique protège nettement ; la plupart des factures sont payées autour de l'échéance, avec une longue traîne."""),
    ("md", """## 2. Variables explicatives, sans fuite d'information
Chaque variable est calculée **à la date d'émission** de la facture, avec ce qu'on savait ce jour-là.
Une facture précédente payée *après* cette date est vue comme « encore impayée » : c'est ce qu'aurait vu
l'équipe de recouvrement. Utiliser le solde final serait une fuite qui gonflerait artificiellement les scores."""),
    ("code", """d = ent.charger()
print(f"{len(prep.VARIABLES)} variables : {len(prep.CATEGORIELLES)} catégorielles, {len(prep.NUMERIQUES)} numériques")
d[prep.NUMERIQUES].describe().T[["mean", "50%", "min", "max"]]"""),
    ("md", """## 3. Découpage temporel
On apprend sur le passé et on teste sur des mois jamais vus, comme en production. Un découpage aléatoire
mélangerait les mois et surestimerait la performance."""),
    ("code", """j = ent.decouper(d)
app, val, test = j["apprentissage"], j["validation"], j["test"]
pd.DataFrame({k: {"factures": len(v), "de": v["periode"].min(), "à": v["periode"].max(),
                  "taux_impaye": v[ent.CIBLE].astype(float).mean()} for k, v in j.items()}).T"""),
    ("md", """## 4. Modèles
1. **Règle métier** : pire retard récent + nombre d'impayés en cours (ce que ferait un analyste sans modèle).
2. **Régression logistique** : référence statistique.
3. **LightGBM** : gradient boosting sur arbres, petite recherche en grille jugée sur la **validation** uniquement.

On ajoute un **oracle** : la vraie probabilité du simulateur, inaccessible dans un vrai projet,
qui donne le plafond atteignable avec ces données."""),
    ("code", """logistique = ent.entrainer_logistique(app)
lgbm, grille = ent.regler_lightgbm(app, val)
pd.DataFrame(grille).sort_values("auc_validation", ascending=False).head()"""),
    ("code", """y = test[ent.CIBLE].to_numpy()
scores = {"Règle métier": ent.score_regle(test),
          "Régression logistique": logistique.predict_proba(test[prep.VARIABLES])[:, 1],
          "LightGBM": lgbm.predict(test[prep.VARIABLES], num_iteration=lgbm.best_iteration)}
verite = pd.read_csv("../data/verite_cachee.csv").set_index("id_facture")["proba_reelle"]
tableau = {nom: ent.metriques(y, s, proba=(nom != "Règle métier")) for nom, s in scores.items()}
tableau["Oracle (plafond)"] = ent.metriques(y, verite.loc[test["id_facture"]].to_numpy())
pd.DataFrame(tableau).T[["auc", "pr_auc", "gini", "ks", "lift_top10", "rappel_top20", "brier"]]"""),
    ("md", """**Lecture honnête** : les deux modèles battent nettement la règle métier, mais LightGBM et la régression
logistique font **jeu égal**. L'écart restant avec l'oracle vient surtout de difficultés soudaines (perte de revenu,
dépense imprévue) qu'aucun historique de paiement ne peut annoncer : il faudrait d'autres sources (recharges,
usage, réclamations). À performance égale, LightGBM est retenu pour ses explications par facture (SHAP),
sa gestion native des valeurs manquantes et des catégories, et parce qu'il absorbera mieux de futures variables
non linéaires ; la régression logistique reste une alternative crédible, plus simple à auditer."""),
    ("code", """resultats = {"gains": {nom: ent.courbe_gains(y, s) for nom, s in scores.items()},
             "calibration": ent.calibration(y, scores["LightGBM"])}
fig.fig_gains(resultats); fig.fig_calibration(resultats); plt.show()"""),
    ("md", """## 5. Interprétation (SHAP)
LightGBM calcule nativement les valeurs SHAP (`pred_contrib=True`) : la contribution de chaque variable
au score de **chaque** facture. C'est ce qui permet d'expliquer une relance à un client ou à un agent."""),
    ("code", """shap = ent.importance_shap(lgbm, test)
fig.fig_importance({"importance": shap.to_dict(orient="records")}); plt.show()
shap.head(10)"""),
    ("code", """# Explication d'une facture à risque élevé
proba = scores["LightGBM"]
i = int(np.argsort(proba)[-50])
ligne = test.iloc[[i]]
contrib = lgbm.predict(ligne[prep.VARIABLES], num_iteration=lgbm.best_iteration, pred_contrib=True)[0]
print(f"Probabilité d'impayé : {proba[i]:.1%} (issue réelle : {int(ligne[ent.CIBLE].iloc[0])})")
pd.DataFrame({"valeur": ligne[prep.VARIABLES].iloc[0].astype(str).to_numpy(), "contribution": contrib[:-1]},
             index=[ent.LIBELLES[v] for v in prep.VARIABLES]).sort_values("contribution", key=abs, ascending=False).head(8)"""),
    ("md", """## 6. Valeur pour le recouvrement
L'équipe peut relancer 20 % des factures chaque mois. Quel ciblage évite le plus d'impayés ?
*Hypothèse explicite* : une relance évite 35 % des impayés qu'elle cible. Ce chiffre doit être **mesuré par un
test A/B** (groupe témoin non relancé) avant toute décision."""),
    ("code", """simulation = ent.simulation_relance(test, scores)
fig.fig_simulation({"simulation_relance": simulation}); plt.show()
pd.DataFrame(simulation)"""),
    ("md", """## 7. Conclusions et suites
- Cibler les relances avec le modèle double environ la part d'impayés atteinte par rapport au hasard,
  et fait mieux que la règle métier.
- Variables décisives : mode de paiement, statut et retard de la facture précédente, régularité passée, validité du contact
  (une facture envoyée par SMS à un numéro invalide n'arrive jamais : action corrective simple et peu coûteuse).
- **Suites** : test A/B des relances ciblées ; modèle de *survie* pour prédire la date probable de paiement
  (trésorerie) ; modèle d'*uplift* pour cibler les clients sensibles à une relance ; nouvelles sources (recharges, usage) ;
  suivi mensuel de la dérive et recalibrage."""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t) for k, t in CELLULES]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    SORTIE.parent.mkdir(exist_ok=True)
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(SORTIE.parent)}}).execute()
    nbf.write(nb, SORTIE)
    print(f"Notebook exécuté -> {SORTIE.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
