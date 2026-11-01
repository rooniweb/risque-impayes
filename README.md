# Risque d'impayés — prédiction avec LightGBM

Prédire, dès l'émission d'une facture télécom, la probabilité qu'elle reste impayée 60 jours après l'échéance.
Données entièrement fictives et reproductibles. Étude complète : [CONTEXTE_PROJET.md](CONTEXTE_PROJET.md).

## Contenu
```
risque_impayes/
├── CONTEXTE_PROJET.md          # étude de cas (problème, démarche, résultats, limites)
├── DICTIONNAIRE_DONNEES.md     # description de chaque colonne
├── requirements.txt
├── src/
│   ├── generer_donnees.py      # simulateur de comptes et de factures (graine fixe)
│   ├── preparer_dataset.py     # variables calculées à la date d'émission (sans fuite)
│   ├── entrainer.py            # règle métier, régression logistique, LightGBM, évaluation, export
│   ├── figures.py              # graphiques
│   └── construire_notebook.py  # construit et exécute le notebook
├── data/                       # comptes, factures, table de modélisation, vérité cachée
├── notebooks/analyse_risque_impayes.ipynb   # l'étude pas à pas, exécutée
├── resultats/                  # resultats.json, modele_lightgbm.txt, figures/
└── demo/                       # page web autonome : résultats + simulateur (modèle évalué dans le navigateur)
```

## Reproduire
Python 3.13 recommandé (LightGBM 4.7.0 plante sous Windows : la version 4.6.0 est figée).
```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -r requirements.txt
```
```bash
.venv/Scripts/python -m src.generer_donnees
```
```bash
.venv/Scripts/python -m src.preparer_dataset
```
```bash
.venv/Scripts/python -m src.entrainer
```
```bash
.venv/Scripts/python -m src.figures
```
```bash
.venv/Scripts/python -m src.construire_notebook
```
Sous Linux ou macOS, remplacer `.venv/Scripts/python` par `.venv/bin/python`.

## Voir la démo
```bash
python -m http.server 8091 --directory demo
```
puis ouvrir http://localhost:8091. La page lit `resultats.json` et `modele_web.json` ; elle vérifie au chargement que
son moteur JavaScript retrouve les probabilités calculées par LightGBM en Python (écart maximal 0,00005).
