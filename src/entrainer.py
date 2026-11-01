"""Entraîne et compare trois approches de prédiction de `risque_impaye`, puis exporte les résultats.

1. Règle métier : « plus le pire retard des 6 dernières factures est grand, plus le risque est élevé ».
2. Régression logistique (référence statistique, interprétable).
3. LightGBM (gradient boosting sur arbres), modèle retenu.

Découpage TEMPOREL (jamais aléatoire) : on apprend sur le passé, on teste sur des mois jamais vus.
Usage : python -m src.entrainer   (écrit resultats/ et demo/)
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preparer_dataset import CATEGORIELLES, CIBLE, NUMERIQUES, VARIABLES

RACINE = Path(__file__).resolve().parent.parent
RESULTATS = RACINE / "resultats"
DEMO = RACINE / "demo"
GRAINE = 42
FIN_APPRENTISSAGE = "2026-01-01"  # apprentissage : factures émises avant cette date
FIN_VALIDATION = "2026-04-01"     # validation (arrêt anticipé, choix du seuil) ; test : à partir de cette date
CAPACITE_RELANCE = 0.20           # l'équipe peut relancer 20 % des factures chaque mois
TAUX_CONVERSION_RELANCE = 0.35    # HYPOTHÈSE : une relance évite 35 % des impayés ciblés (à mesurer par test A/B)

PARAMETRES_LGBM = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=200,
                       feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                       verbose=-1, seed=GRAINE)

LIBELLES = {
    "type_client": "Type de client", "region": "Région", "univers": "Univers de l'offre",
    "mode_paiement": "Mode de paiement habituel", "canal_facture": "Canal de réception de la facture",
    "nb_lignes": "Nombre de lignes", "contact_valide": "Contact valide", "anciennete_mois": "Ancienneté (mois)",
    "mois_emission": "Mois d'émission", "montant": "Montant de la facture", "ratio_montant_6m": "Montant / moyenne 6 mois",
    "nb_factures_passees": "Nombre de factures passées", "retard_moyen_6m": "Retard moyen 6 mois (j)",
    "retard_max_6m": "Pire retard 6 mois (j)", "retard_derniere_facture": "Retard facture précédente (j)",
    "tendance_retard": "Tendance du retard (j)", "part_a_temps_12m": "Part payée à temps 12 mois",
    "nb_impayes_en_cours": "Factures impayées en cours", "montant_impaye_en_cours": "Montant impayé en cours",
    "derniere_facture_payee": "Facture précédente déjà payée", "retard_habituel_12m": "Retard habituel 12 mois (j)",
    "jours_depuis_dernier_paiement": "Jours depuis le dernier paiement", "nb_impayes_60j_12m": "Impayés à 60 j sur 12 mois",
}


# ---------- Données ----------
def charger() -> pd.DataFrame:
    d = pd.read_csv(RACINE / "data" / "dataset_modelisation.csv", parse_dates=["date_emission", "date_echeance"],
                    dtype={CIBLE: "Int64"})
    for c in CATEGORIELLES:
        d[c] = d[c].astype("category")
    return d


def decouper(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    connu = d[d[CIBLE].notna()].copy()
    connu[CIBLE] = connu[CIBLE].astype(int)
    return {
        "apprentissage": connu[connu["date_emission"] < FIN_APPRENTISSAGE],
        "validation": connu[(connu["date_emission"] >= FIN_APPRENTISSAGE) & (connu["date_emission"] < FIN_VALIDATION)],
        "test": connu[connu["date_emission"] >= FIN_VALIDATION],
        "a_scorer": d[d[CIBLE].isna()].copy(),  # factures récentes dont l'issue est inconnue
    }


# ---------- Modèles ----------
def score_regle(x: pd.DataFrame) -> np.ndarray:
    return x["retard_max_6m"].fillna(0).to_numpy() + 30 * x["nb_impayes_en_cours"].to_numpy()


def entrainer_logistique(app: pd.DataFrame):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUMERIQUES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORIELLES)])
    modele = make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))
    return modele.fit(app[VARIABLES], app[CIBLE])


GRILLE_LGBM = [dict(num_leaves=f, min_child_samples=m) for f in (7, 15, 31) for m in (50, 200, 500)]


def entrainer_lightgbm(app: pd.DataFrame, val: pd.DataFrame, parametres: dict | None = None) -> lgb.Booster:
    jeu_app = lgb.Dataset(app[VARIABLES], app[CIBLE].to_numpy(dtype=float), categorical_feature=CATEGORIELLES)
    jeu_val = lgb.Dataset(val[VARIABLES], val[CIBLE].to_numpy(dtype=float), reference=jeu_app)
    return lgb.train({**PARAMETRES_LGBM, **(parametres or {}), "metric": "auc"}, jeu_app, num_boost_round=3000,
                     valid_sets=[jeu_val], callbacks=[lgb.early_stopping(100, verbose=False)])


def regler_lightgbm(app: pd.DataFrame, val: pd.DataFrame) -> tuple[lgb.Booster, list[dict]]:
    """Petite recherche en grille, jugée sur l'AUC de VALIDATION (le test n'est jamais regardé)."""
    essais = []
    for g in GRILLE_LGBM:
        m = entrainer_lightgbm(app, val, g)
        essais.append({**g, "nb_arbres": m.best_iteration, "auc_validation": round(m.best_score["valid_0"]["auc"], 4),
                       "modele": m})
    meilleur = max(essais, key=lambda e: e["auc_validation"])
    return meilleur["modele"], [{k: v for k, v in e.items() if k != "modele"} for e in essais]


# ---------- Évaluation ----------
def metriques(y: np.ndarray, score: np.ndarray, proba: bool = True) -> dict:
    ordre = np.argsort(-score)
    y_trie = y[ordre]
    res = {"auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score)}
    res["gini"] = 2 * res["auc"] - 1
    fpr, tpr, _ = roc_curve(y, score)
    res["ks"] = float(np.max(tpr - fpr))
    for part in (0.10, 0.20):
        k = int(len(y) * part)
        res[f"rappel_top{int(part * 100)}"] = y_trie[:k].sum() / y.sum()
        res[f"lift_top{int(part * 100)}"] = y_trie[:k].mean() / y.mean()
    if proba:
        res["brier"] = brier_score_loss(y, score)
    return {k: round(float(v), 4) for k, v in res.items()}


def seuil_f1(y: np.ndarray, proba: np.ndarray) -> float:
    precision, rappel, seuils = precision_recall_curve(y, proba)
    f1 = 2 * precision * rappel / np.maximum(precision + rappel, 1e-9)
    return float(seuils[np.argmax(f1[:-1])])


def importance_shap(modele: lgb.Booster, x: pd.DataFrame) -> pd.DataFrame:
    """Valeurs SHAP natives de LightGBM (pred_contrib) : contribution de chaque variable au log-odds."""
    contrib = modele.predict(x[VARIABLES], pred_contrib=True)[:, :-1]
    return pd.DataFrame({"variable": VARIABLES, "libelle": [LIBELLES[v] for v in VARIABLES],
                         "shap_moyen_abs": np.abs(contrib).mean(axis=0)}).sort_values("shap_moyen_abs", ascending=False)


def simulation_relance(test: pd.DataFrame, scores: dict[str, np.ndarray]) -> list[dict]:
    """Montant d'impayés évité si l'on relance les 20 % de factures les plus risquées de chaque mois."""
    lignes = []
    total_impaye = test.loc[test[CIBLE] == 1, "montant"].sum()
    for nom, score in scores.items():
        t = test.assign(score=score)
        cible = t.groupby("periode", group_keys=False).apply(
            lambda g: g.nlargest(int(len(g) * CAPACITE_RELANCE), "score"))
        impaye_cible = cible.loc[cible[CIBLE] == 1, "montant"].sum()
        lignes.append({"strategie": nom, "factures_relancees": len(cible),
                       "impayes_captes_pct": round(100 * impaye_cible / total_impaye, 1),
                       "montant_evite_fcfa": int(TAUX_CONVERSION_RELANCE * impaye_cible)})
    hasard = int(TAUX_CONVERSION_RELANCE * CAPACITE_RELANCE * total_impaye)
    lignes.insert(0, {"strategie": "Relance au hasard", "factures_relancees": lignes[0]["factures_relancees"],
                      "impayes_captes_pct": round(100 * CAPACITE_RELANCE, 1), "montant_evite_fcfa": hasard})
    return lignes


def calibration(y: np.ndarray, proba: np.ndarray, nb: int = 10) -> list[dict]:
    tranches = pd.qcut(proba, nb, labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": proba, "t": tranches}).groupby("t").agg(predit=("p", "mean"), observe=("y", "mean"))
    return [{"predit": round(float(a), 4), "observe": round(float(b), 4)} for a, b in df.itertuples(index=False)]


# ---------- Export pour la démo web ----------
def exporter_modele_web(modele: lgb.Booster, x_ref: pd.DataFrame) -> dict:
    """Arbres sous forme compacte, évalués en JavaScript dans le navigateur."""
    dump = modele.dump_model()
    categories = {c: list(x_ref[c].cat.categories) for c in CATEGORIELLES}

    def noeud(n):
        if "leaf_value" in n:
            return round(n["leaf_value"], 6)
        d = {"f": n["split_feature"], "g": noeud(n["left_child"]), "d": noeud(n["right_child"]),
             "m": n["missing_type"], "dl": bool(n["default_left"])}
        if n["decision_type"] == "==":
            d["c"] = [int(v) for v in str(n["threshold"]).split("||")]
        else:
            d["s"] = n["threshold"]
        return d

    numeriques = {v: {"min": float(x_ref[v].quantile(0.01)), "max": float(x_ref[v].quantile(0.99)),
                      "mediane": float(x_ref[v].median())} for v in NUMERIQUES}
    return {"variables": VARIABLES, "categorielles": CATEGORIELLES, "categories": categories, "libelles": LIBELLES,
            "numeriques": numeriques, "arbres": [noeud(t["tree_structure"]) for t in dump["tree_info"]]}


def courbe_gains(y: np.ndarray, score: np.ndarray, pas: int = 20) -> list[float]:
    y_trie = y[np.argsort(-score)]
    cumul = np.cumsum(y_trie) / y.sum()
    return [round(float(cumul[int(len(y) * k / pas) - 1]), 4) for k in range(1, pas + 1)]


def main() -> None:
    RESULTATS.mkdir(exist_ok=True)
    DEMO.mkdir(exist_ok=True)
    d = charger()
    j = decouper(d)
    app, val, test = j["apprentissage"], j["validation"], j["test"]
    y = test[CIBLE].to_numpy()

    logistique = entrainer_logistique(app)
    lgbm, grille = regler_lightgbm(app, val)
    scores = {
        "Règle métier": score_regle(test),
        "Régression logistique": logistique.predict_proba(test[VARIABLES])[:, 1],
        "LightGBM": lgbm.predict(test[VARIABLES], num_iteration=lgbm.best_iteration),
    }
    comparaison = {nom: metriques(y, s, proba=(nom != "Règle métier")) for nom, s in scores.items()}
    # Plafond : AUC obtenue avec la vraie probabilité du simulateur (inaccessible dans un vrai projet)
    verite = pd.read_csv(RACINE / "data" / "verite_cachee.csv").set_index("id_facture")["proba_reelle"]
    oracle = metriques(y, verite.loc[test["id_facture"]].to_numpy())
    proba = scores["LightGBM"]
    seuil = seuil_f1(val[CIBLE].to_numpy(), lgbm.predict(val[VARIABLES], num_iteration=lgbm.best_iteration))
    tn, fp, fn, tp = confusion_matrix(y, (proba >= seuil).astype(int)).ravel()
    shap = importance_shap(lgbm, test)

    # Portefeuille courant : factures récentes à issue inconnue, classées en niveaux de risque
    a_scorer = j["a_scorer"]
    a_scorer["proba"] = lgbm.predict(a_scorer[VARIABLES], num_iteration=lgbm.best_iteration)
    niveaux = pd.cut(a_scorer["proba"], [0, 0.10, 0.25, 0.50, 1], labels=["Faible", "Modéré", "Élevé", "Très élevé"])
    portefeuille = a_scorer.groupby(niveaux, observed=False).agg(factures=("id_facture", "size"),
                                                                 montant_fcfa=("montant", "sum"))

    taux_mensuel = d[d[CIBLE].notna()].assign(y=lambda x: x[CIBLE].astype(int)).groupby("periode")["y"].mean()
    delais = d["delai_paiement_jours"].dropna().clip(-15, 120)
    resultats = {
        "donnees": {"comptes": int(d["id_compte"].nunique()), "factures": len(d),
                    "factures_cible_connue": int(d[CIBLE].notna().sum()),
                    "taux_impaye": round(float(d[CIBLE].dropna().astype(int).mean()), 4),
                    "apprentissage": len(app), "validation": len(val), "test": len(test),
                    "periode_test": f"{test['periode'].min()} à {test['periode'].max()}",
                    "nb_mois_test": int(test["periode"].nunique())},
        "comparaison": comparaison,
        "oracle": oracle,
        "grille_lightgbm": grille,
        "lightgbm": {"nb_arbres": lgbm.best_iteration, "parametres": {k: v for k, v in lgbm.params.items()
                                                                    if k not in ("num_iterations", "categorical_column", "metric")}, "seuil_f1": round(seuil, 4),
                     "matrice_confusion": {"vrais_negatifs": int(tn), "faux_positifs": int(fp),
                                           "faux_negatifs": int(fn), "vrais_positifs": int(tp)},
                     "precision": round(tp / (tp + fp), 4), "rappel": round(tp / (tp + fn), 4),
                     "f1": round(f1_score(y, (proba >= seuil).astype(int)), 4)},
        "importance": shap.round(4).to_dict(orient="records"),
        "calibration": calibration(y, proba),
        "gains": {nom: courbe_gains(y, s) for nom, s in scores.items()},
        "simulation_relance": simulation_relance(test, scores),
        "hypotheses_simulation": {"capacite_relance": CAPACITE_RELANCE, "taux_conversion_relance": TAUX_CONVERSION_RELANCE},
        "portefeuille_courant": [{"niveau": str(k), "factures": int(v.factures), "montant_fcfa": int(v.montant_fcfa)}
                                 for k, v in portefeuille.iterrows()],
        "taux_impaye_mensuel": {k: round(float(v), 4) for k, v in taux_mensuel.items()},
        "taux_impaye_par_mode": {k: round(float(v), 4) for k, v in
                                 d[d[CIBLE].notna()].groupby("mode_paiement", observed=True)[CIBLE].mean().items()},
        "histogramme_delais": np.histogram(delais, bins=np.arange(-15, 125, 5))[0].tolist(),
    }
    # Exemples expliqués (valeurs SHAP par facture) pour la démo
    echantillon = test.sample(400, random_state=GRAINE)
    p_ech = lgbm.predict(echantillon[VARIABLES], num_iteration=lgbm.best_iteration)
    choisis = echantillon.iloc[np.argsort(p_ech)[[5, 120, 250, 330, 385, 398]]]
    contrib = lgbm.predict(choisis[VARIABLES], num_iteration=lgbm.best_iteration, pred_contrib=True)
    resultats["exemples"] = [{
        "id_facture": int(r.id_facture), "valeurs": {v: (str(getattr(r, v)) if v in CATEGORIELLES else
                                                         (None if pd.isna(getattr(r, v)) else float(getattr(r, v))))
                                                     for v in VARIABLES},
        "proba": round(float(1 / (1 + np.exp(-c.sum()))), 4), "base": round(float(c[-1]), 4),
        "contributions": {v: round(float(x), 4) for v, x in zip(VARIABLES, c[:-1])},
        "issue_reelle": int(r.risque_impaye)} for r, c in zip(choisis.itertuples(index=False), contrib)]

    lgbm.save_model(str(RESULTATS / "modele_lightgbm.txt"), num_iteration=lgbm.best_iteration)
    (RESULTATS / "resultats.json").write_text(json.dumps(resultats, ensure_ascii=False, indent=1), encoding="utf-8")
    web = exporter_modele_web(lgbm, app)
    # Contrôle : quelques probabilités de référence que la démo JavaScript doit retrouver
    web["controle"] = [{"valeurs": e["valeurs"], "proba": e["proba"]} for e in resultats["exemples"]]
    (DEMO / "modele_web.json").write_text(json.dumps(web, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (DEMO / "resultats.json").write_text(json.dumps(resultats, ensure_ascii=False), encoding="utf-8")

    print(pd.DataFrame({**comparaison, "Oracle (vraie probabilité)": oracle}).T.to_string())
    print(pd.DataFrame(grille).to_string(index=False))
    print(f"\nLightGBM : {lgbm.best_iteration} arbres, seuil F1 {seuil:.3f}, "
          f"précision {resultats['lightgbm']['precision']:.1%}, rappel {resultats['lightgbm']['rappel']:.1%}")
    print(pd.DataFrame(resultats["simulation_relance"]).to_string(index=False))
    print(shap.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
