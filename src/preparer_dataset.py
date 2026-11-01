"""Construit la table de modélisation : une ligne par facture, variables calculées À LA DATE D'ÉMISSION.

Règle anti-fuite : pour la facture émise le jour E, on n'utilise que ce qu'on savait le jour E.
Une facture précédente payée après E est donc vue comme « encore impayée, avec X jours de retard au jour E ».

Usage : python -m src.preparer_dataset   (écrit data/dataset_modelisation.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd

DOSSIER = Path(__file__).resolve().parent.parent / "data"

CATEGORIELLES = ["type_client", "region", "univers", "mode_paiement", "canal_facture"]
NUMERIQUES = ["nb_lignes", "contact_valide", "anciennete_mois", "mois_emission", "montant", "ratio_montant_6m",
              "nb_factures_passees", "retard_moyen_6m", "retard_max_6m", "retard_derniere_facture",
              "tendance_retard", "part_a_temps_12m", "nb_impayes_en_cours", "montant_impaye_en_cours",
              "derniere_facture_payee", "retard_habituel_12m", "jours_depuis_dernier_paiement", "nb_impayes_60j_12m"]
VARIABLES = CATEGORIELLES + NUMERIQUES
CIBLE = "risque_impaye"


def charger_brut() -> tuple[pd.DataFrame, pd.DataFrame]:
    comptes = pd.read_csv(DOSSIER / "comptes.csv", parse_dates=["date_ouverture"])
    factures = pd.read_csv(DOSSIER / "factures.csv", parse_dates=["date_emission", "date_echeance", "date_solde"],
                           dtype={"risque_impaye": "Int64", "delai_paiement_jours": "Int64"})
    return comptes, factures


def _historique_compte(g: pd.DataFrame) -> pd.DataFrame:
    """Variables d'historique pour les factures d'un compte, triées par date d'émission."""
    emission = g["date_emission"].to_numpy()
    echeance = g["date_echeance"].to_numpy()
    solde = g["date_solde"].to_numpy()
    montant = g["montant"].to_numpy(dtype=float)
    n = len(g)
    sortie = {k: np.full(n, np.nan) for k in ["ratio_montant_6m", "retard_moyen_6m", "retard_max_6m",
                                               "retard_derniere_facture", "tendance_retard", "part_a_temps_12m",
                                               "nb_impayes_en_cours", "montant_impaye_en_cours",
                                               "derniere_facture_payee", "retard_habituel_12m",
                                               "jours_depuis_dernier_paiement", "nb_impayes_60j_12m"]}
    sortie["nb_factures_passees"] = np.arange(n, dtype=float)
    for i in range(n):
        if i == 0:
            sortie["ratio_montant_6m"][i] = 1.0
            sortie["nb_impayes_en_cours"][i] = 0
            sortie["montant_impaye_en_cours"][i] = 0
            sortie["nb_impayes_60j_12m"][i] = 0
            continue
        e = emission[i]
        passees = slice(0, i)
        paye_connu = ~np.isnat(solde[passees]) & (solde[passees] <= e)
        # Retard observé au jour E : délai réel si payée, sinon retard courant (borne basse)
        retard = np.where(paye_connu, (solde[passees] - echeance[passees]) / np.timedelta64(1, "D"),
                          (e - echeance[passees]) / np.timedelta64(1, "D"))
        sortie["ratio_montant_6m"][i] = montant[i] / montant[max(0, i - 6):i].mean()
        sortie["retard_moyen_6m"][i] = retard[-6:].mean()
        sortie["retard_max_6m"][i] = retard[-6:].max()
        sortie["retard_derniere_facture"][i] = retard[-1]
        if i >= 6:
            sortie["tendance_retard"][i] = retard[-3:].mean() - retard[-6:-3].mean()
        sortie["part_a_temps_12m"][i] = (paye_connu[-12:] & (retard[-12:] <= 0)).mean()
        sortie["nb_impayes_en_cours"][i] = (~paye_connu).sum()
        sortie["montant_impaye_en_cours"][i] = montant[passees][~paye_connu].sum()
        sortie["derniere_facture_payee"][i] = float(paye_connu[-1])
        # Retard habituel : délais réels des seules factures payées (le « rythme » du payeur)
        payes_12 = retard[-12:][paye_connu[-12:]]
        if len(payes_12):
            sortie["retard_habituel_12m"][i] = np.median(payes_12)
        if paye_connu.any():
            sortie["jours_depuis_dernier_paiement"][i] = (e - solde[passees][paye_connu].max()) / np.timedelta64(1, "D")
        # Impayés à 60 jours déjà constatés au jour E (échéance passée de plus de 60 jours, non soldée à temps)
        constat = (e - echeance[passees]) / np.timedelta64(1, "D") > 60
        sortie["nb_impayes_60j_12m"][i] = (constat & (retard > 60))[-12:].sum()
    return pd.DataFrame(sortie, index=g.index)


def construire(comptes: pd.DataFrame, factures: pd.DataFrame) -> pd.DataFrame:
    f = factures.sort_values(["id_compte", "date_emission"]).copy()
    historique = f.groupby("id_compte", group_keys=False)[["date_emission", "date_echeance", "date_solde", "montant"]] \
        .apply(_historique_compte)
    f = f.join(historique).merge(comptes, on="id_compte", how="left")
    f["anciennete_mois"] = ((f["date_emission"] - f["date_ouverture"]).dt.days / 30.4).round(1)
    f["mois_emission"] = f["date_emission"].dt.month
    colonnes = ["id_facture", "id_compte", "periode", "date_emission", "date_echeance", "montant"] + \
               [v for v in VARIABLES if v != "montant"] + ["delai_paiement_jours", CIBLE]
    return f[colonnes].sort_values("id_facture").reset_index(drop=True)


def main() -> None:
    comptes, factures = charger_brut()
    d = construire(comptes, factures)
    d.to_csv(DOSSIER / "dataset_modelisation.csv", index=False, date_format="%Y-%m-%d", float_format="%.4g")
    print(f"{len(d)} lignes, {len(VARIABLES)} variables explicatives -> data/dataset_modelisation.csv")


if __name__ == "__main__":
    main()
