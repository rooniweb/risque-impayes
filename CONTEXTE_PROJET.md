# Prédire le risque d'impayé des factures télécom (LightGBM)

> Étude de cas pour un portfolio data analyst / data scientist.
> Données **entièrement simulées** (opérateur télécom imaginaire d'Afrique de l'Ouest), générées par un script
> à graine fixe : chaque chiffre ci-dessous se reproduit à l'identique.

## Problème métier
Une équipe de recouvrement ne peut pas relancer toutes les factures. Elle relance tard (une fois le retard constaté
dans la balance âgée) et au hasard des listes. **Question** : dès l'émission d'une facture, peut-on estimer la
probabilité qu'elle reste impayée 60 jours après son échéance, pour relancer d'abord celles qui en ont besoin ?

## Données
- 5 000 comptes postpayés, ≈ 100 000 factures mensuelles de janvier 2024 à septembre 2026.
- Pour chaque facture : montant, échéance, date de paiement complet → **délai de paiement**.
- **Cible binaire `risque_impaye`** : 1 si la facture n'est pas soldée 60 jours après l'échéance (17,2 % des factures).
- Le simulateur reproduit des mécanismes réalistes : profil du payeur, périodes de difficulté qui durent quelques mois,
  effets de calendrier (Tabaski, rentrée scolaire), factures anormalement élevées, contacts de facturation invalides,
  et des **interactions** (une difficulté pèse plus sur un particulier qui paie en espèces que sur un compte en prélèvement).

## Démarche
1. **Exploration** : distribution des délais, saisonnalité (pics d'impayés les mois de Tabaski), effet du mode de paiement
   (5 % d'impayés en prélèvement contre 24 % en espèces).
2. **Feature engineering sans fuite** : 23 variables calculées **à la date d'émission**, avec ce que l'on savait ce jour-là
   (retards passés, statut de la facture précédente, régularité sur 12 mois, impayés en cours, tendance, ratio de montant…).
   Une facture précédente payée *après* cette date est vue comme encore impayée.
3. **Découpage temporel** : apprentissage sur 2024-2025, validation (arrêt anticipé, choix du seuil, recherche en grille)
   puis **test sur 5 mois jamais vus** (mars à juillet 2026).
4. **Trois approches comparées** : règle métier, régression logistique, LightGBM.
5. **Plafond théorique** : comme les données sont simulées, la vraie probabilité est connue ; elle donne l'AUC maximale
   atteignable (« oracle ») et sépare ce qui est perdu par le modèle de ce qui est imprévisible.
6. **Interprétation** : valeurs SHAP natives de LightGBM, globales et par facture.
7. **Valeur métier** : simulation d'une campagne où l'équipe relance 20 % des factures chaque mois.

## Résultats (jeu de test, 14 050 factures)
| Modèle | AUC | PR-AUC | Lift top 10 % | Part des factures impayées dans les 20 % les plus risquées |
|---|---|---|---|---|
| Règle métier (pire retard + impayés en cours) | 0,650 | 0,280 | × 2,1 | 36 % |
| Régression logistique | 0,728 | 0,363 | × 2,7 | 45 % |
| **LightGBM (retenu)** | **0,725** | **0,364** | **× 2,8** | **45 %** |
| *Oracle (vraie probabilité, plafond)* | *0,797* | *0,517* | *× 3,7* | *55 %* |

- **Calibration** : Brier 0,122 ; probabilités prédites alignées sur les taux observés par décile.
- **Seuil** choisi sur la validation (F1 maximal, 0,22) : précision 34 %, rappel 50 % sur le test.
- **Campagne simulée** (hypothèse : une relance évite 35 % des impayés ciblés) : ≈ 8,3 M FCFA d'impayés évités sur
  5 mois avec le modèle (40 % du montant impayé ciblé), contre 6,9 M avec la règle métier et 4,2 M en relançant au hasard,
  soit **× 2 par rapport au hasard**. La régression logistique fait le même score.

### Lecture honnête
LightGBM et la régression logistique font **jeu égal** et battent nettement la règle métier. L'écart avec l'oracle vient
surtout de difficultés soudaines (perte de revenu, dépense imprévue) qu'aucun historique de paiement ne peut annoncer :
plus de feature engineering n'y a rien changé, il faudrait d'autres sources (recharges, usage, réclamations).
À performance égale, LightGBM est retenu pour ses **explications par facture** (SHAP), sa gestion native des valeurs
manquantes et des catégories, et sa capacité à absorber de futures variables non linéaires. La régression logistique
reste une alternative crédible, plus simple à auditer.

## Ce que le modèle apprend (SHAP)
Mode de paiement habituel, statut et retard de la facture précédente, retard moyen récent, validité du contact,
type de client, ancienneté. Un levier ressort immédiatement : **une facture envoyée par SMS à un numéro invalide n'arrive
jamais**, d'où un risque nettement plus élevé. Fiabiliser les contacts est une action simple et peu coûteuse.

## Du score à l'action
| Niveau | Probabilité | Action de recouvrement |
|---|---|---|
| Faible | < 10 % | SMS automatique à l'échéance |
| Modéré | 10 à 25 % | Rappel personnalisé 3 jours avant l'échéance, lien de paiement mobile money |
| Élevé | 25 à 50 % | Appel d'un agent dès l'échéance, vérification du contact |
| Très élevé | ≥ 50 % | Appel prioritaire, proposition d'échéancier avant que la dette ne s'accumule |

Autres usages : prévision des encaissements (Σ montant × (1 − p)), provisions pour créances douteuses,
comparaison des agences à risque de portefeuille égal.

## Limites et suites
- L'efficacité d'une relance (35 %) est une **hypothèse** : à mesurer par un **test A/B** avec groupe témoin.
- Prédire n'est pas convaincre : un modèle d'**uplift** ciblerait les clients sensibles à une relance.
- Un modèle de **survie** prédirait la date probable de paiement (trésorerie).
- Boucle de rétroaction : une fois les relances ciblées, les clients relancés paient davantage ; à prendre en compte au réentraînement.
- Éthique : aucune variable sensible (âge, sexe, quartier) ; le score oriente les relances, il ne déclenche aucune action automatique.

## Stack
Python 3.13 · pandas · NumPy · scikit-learn · LightGBM · SHAP (natif LightGBM) · matplotlib · Jupyter ·
démo web en HTML/JavaScript natif (le modèle LightGBM est exporté en JSON et évalué dans le navigateur).
