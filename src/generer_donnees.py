"""Génère les données FICTIVES d'un opérateur télécom imaginaire : comptes postpayés et factures mensuelles.

Le comportement de paiement suit un mécanisme caché, volontairement réaliste :
- un risque de base propre à chaque compte (mode de paiement, contact valide, ancienneté, effet individuel) ;
- un état de « difficulté » qui apparaît et disparaît dans le temps (chaîne de Markov), d'où l'intérêt de l'historique ;
- des effets de calendrier (Tabaski, rentrée scolaire, fin d'année) et des factures anormalement élevées.
Le modèle ne voit jamais ce mécanisme : il doit le retrouver à partir de l'historique observable.

Usage : python -m src.generer_donnees   (écrit data/comptes.csv et data/factures.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd

GRAINE = 42
NB_COMPTES = 5000
DEBUT = pd.Timestamp("2024-01-01")
DERNIERE_PERIODE = pd.Timestamp("2026-09-01")
DATE_OBSERVATION = pd.Timestamp("2026-10-31")  # « aujourd'hui » : on ne connaît rien au-delà
DOSSIER = Path(__file__).resolve().parent.parent / "data"

REGIONS = {"Dakar": 0.45, "Thiès": 0.13, "Touba": 0.10, "Kaolack": 0.08, "Saint-Louis": 0.08,
           "Ziguinchor": 0.07, "Tambacounda": 0.05, "Louga": 0.04}
EFFET_REGION = {"Dakar": -0.10, "Thiès": 0.0, "Touba": 0.25, "Kaolack": 0.15, "Saint-Louis": 0.0,
                "Ziguinchor": 0.20, "Tambacounda": 0.30, "Louga": 0.10}
# Mois de Tabaski (dépenses familiales importantes), rentrée scolaire et fin d'année
EFFET_MOIS = {"2024-06", "2025-06", "2026-05"}
EFFET_MOIS_RECURRENT = {10: 0.25, 12: 0.15}


def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def generer_comptes(r: np.random.Generator) -> pd.DataFrame:
    n = NB_COMPTES
    type_client = r.choice(["PARTICULIER", "PROFESSIONNEL", "ENTREPRISE"], n, p=[0.78, 0.14, 0.08])
    region = r.choice(list(REGIONS), n, p=list(REGIONS.values()))
    univers = np.where(type_client == "PARTICULIER",
                       r.choice(["MOBILE", "FIBRE", "BOX_4G", "FIXE"], n, p=[0.50, 0.25, 0.18, 0.07]),
                       r.choice(["MOBILE", "FIBRE", "BOX_4G", "FIXE"], n, p=[0.45, 0.35, 0.05, 0.15]))
    nb_lignes = np.where(type_client == "PARTICULIER", r.integers(1, 3, n),
                         np.where(type_client == "PROFESSIONNEL", r.integers(1, 5, n), r.integers(2, 21, n)))
    mode_paiement = np.where(type_client == "ENTREPRISE",
                             r.choice(["VIREMENT", "PRELEVEMENT", "MOBILE_MONEY", "ESPECES"], n, p=[0.5, 0.25, 0.15, 0.10]),
                             r.choice(["MOBILE_MONEY", "ESPECES", "PRELEVEMENT", "CARTE", "VIREMENT"], n,
                                      p=[0.50, 0.30, 0.10, 0.06, 0.04]))
    canal_facture = r.choice(["SMS", "EMAIL", "PAPIER"], n, p=[0.52, 0.24, 0.24])
    contact_valide = r.random(n) < 0.86
    ouverture = pd.to_datetime("2015-01-01") + pd.to_timedelta(
        (r.beta(2.2, 1.3, n) * (pd.Timestamp("2026-06-30") - pd.Timestamp("2015-01-01")).days).astype(int), unit="D")
    prix_univers = {"MOBILE": 9000, "FIBRE": 24000, "BOX_4G": 15000, "FIXE": 7000}
    montant_base = np.array([prix_univers[u] for u in univers]) * (1 + 0.6 * (nb_lignes - 1)) * r.lognormal(0, 0.25, n)

    # Risque de base (caché) : ce que le modèle devra approcher sans le voir
    logit = (-2.6
             + np.select([mode_paiement == "PRELEVEMENT", mode_paiement == "ESPECES", mode_paiement == "VIREMENT"],
                         [-1.3, 0.35, -0.3], 0.0)
             # Interaction : une facture envoyée par SMS ou e-mail à un contact invalide n'arrive jamais
             + np.where(contact_valide, 0.0, np.where(canal_facture == "PAPIER", 0.30, 1.40))
             + np.where(canal_facture == "PAPIER", 0.30, 0.0)
             + np.where(type_client == "PARTICULIER", 0.20, np.where(type_client == "ENTREPRISE", -0.35, 0.0))
             + np.array([EFFET_REGION[x] for x in region])
             + r.normal(0, 0.7, n))
    return pd.DataFrame({
        "id_compte": np.arange(1, n + 1), "type_client": type_client, "region": region, "univers": univers,
        "nb_lignes": nb_lignes, "mode_paiement": mode_paiement, "canal_facture": canal_facture,
        "contact_valide": contact_valide.astype(int), "date_ouverture": ouverture.normalize(),
        "cache_montant_base": montant_base.round(-2), "cache_logit_base": logit})


def generer_factures(r: np.random.Generator, comptes: pd.DataFrame) -> pd.DataFrame:
    periodes = pd.date_range(DEBUT, DERNIERE_PERIODE, freq="MS")
    lignes = []
    id_facture = 0
    for c in comptes.itertuples(index=False):
        premiere = max(DEBUT, (c.date_ouverture + pd.offsets.MonthBegin(1)).normalize())
        difficulte = r.random() < 0.08
        # Effets non linéaires et interactions (le mécanisme réel n'est pas une somme d'effets indépendants) :
        # une difficulté pèse plus sur un particulier payant en espèces ou mobile money qu'avec un prélèvement ;
        # une facture anormalement élevée ne gêne vraiment que les petits budgets ; la Tabaski pèse plus hors Dakar.
        poids_difficulte = (1.0 + 1.0 * (c.type_client == "PARTICULIER")
                            + 0.6 * (c.mode_paiement in ("ESPECES", "MOBILE_MONEY")) - 1.0 * (c.mode_paiement == "PRELEVEMENT"))
        poids_choc = 1.1 if c.type_client == "PARTICULIER" and c.univers in ("MOBILE", "BOX_4G") else 0.2
        poids_fete = 0.25 if c.region == "Dakar" else 0.75
        habitude = r.normal(2 * c.cache_logit_base + 5, 5)  # retard habituel du payeur, en jours
        for periode in periodes[periodes >= premiere]:
            # Entrée / sortie de difficulté : les mauvaises périodes durent quelques mois
            if difficulte:
                difficulte = r.random() > 0.22
            else:
                difficulte = r.random() < 0.02 + 0.05 * sigmoide(c.cache_logit_base + 1)
            anciennete_mois = (periode - c.date_ouverture).days / 30.4
            montant = c.cache_montant_base * r.lognormal(0, 0.12)
            choc = r.random() < 0.04
            if choc:
                montant *= r.uniform(1.6, 2.6)
            cle = periode.strftime("%Y-%m")
            logit = (c.cache_logit_base + poids_difficulte * difficulte + poids_choc * choc
                     + (0.8 if anciennete_mois < 6 else 0.4 if anciennete_mois < 12 else 0.0)
                     + poids_fete * (cle in EFFET_MOIS) + EFFET_MOIS_RECURRENT.get(periode.month, 0.0))
            p = sigmoide(logit)
            emission = periode + pd.offsets.MonthBegin(1) + pd.Timedelta(days=2)
            echeance = emission + pd.Timedelta(days=15)
            if r.random() < p:  # impayé à 60 jours
                delai = int(r.integers(61, 241)) if r.random() < 0.5 else None
            else:
                delai = int(np.clip(round(r.normal(habitude + 25 * p + 12 * difficulte, 6)), -15, 60))
            date_solde = echeance + pd.Timedelta(days=delai) if delai is not None else pd.NaT
            if pd.notna(date_solde) and date_solde > DATE_OBSERVATION:
                date_solde = pd.NaT  # pas encore payée à la date d'observation
            id_facture += 1
            lignes.append((id_facture, c.id_compte, cle, emission, echeance, int(round(montant, -2)), date_solde, p))
            # Résiliation : plus fréquente après un impayé, rare sinon
            if (delai is None or delai > 60) and r.random() < 0.12:
                break
            if r.random() < 0.004:
                break
    f = pd.DataFrame(lignes, columns=["id_facture", "id_compte", "periode", "date_emission", "date_echeance",
                                      "montant", "date_solde", "proba_reelle"])
    connu = f["date_echeance"] + pd.Timedelta(days=60) <= DATE_OBSERVATION
    f["delai_paiement_jours"] = (f["date_solde"] - f["date_echeance"]).dt.days
    # Cible : 1 si la facture n'est pas soldée 60 jours après l'échéance ; vide si on ne le sait pas encore
    impaye = f["date_solde"].isna() | (f["delai_paiement_jours"] > 60)
    f["risque_impaye"] = pd.array(np.where(connu, impaye.astype(int), 0), dtype="Int64")
    f.loc[~connu, "risque_impaye"] = pd.NA
    f["delai_paiement_jours"] = f["delai_paiement_jours"].astype("Int64")
    return f


def main() -> None:
    r = np.random.default_rng(GRAINE)
    comptes = generer_comptes(r)
    factures = generer_factures(r, comptes)
    DOSSIER.mkdir(exist_ok=True)
    comptes.drop(columns=["cache_montant_base", "cache_logit_base"]).to_csv(DOSSIER / "comptes.csv", index=False)
    # La vraie probabilité n'est connue que parce que les données sont simulées : elle sert uniquement à mesurer
    # le plafond atteignable (AUC « oracle »), jamais à l'apprentissage.
    factures[["id_facture", "proba_reelle"]].round(4).to_csv(DOSSIER / "verite_cachee.csv", index=False)
    factures.drop(columns="proba_reelle").to_csv(DOSSIER / "factures.csv", index=False, date_format="%Y-%m-%d")
    connues = factures["risque_impaye"].notna()
    print(f"{len(comptes)} comptes, {len(factures)} factures "
          f"({connues.sum()} à cible connue, taux d'impayé {factures.loc[connues, 'risque_impaye'].mean():.1%})")


if __name__ == "__main__":
    main()
