# Dictionnaire des données (toutes fictives)

## `data/comptes.csv` — 5 000 comptes postpayés
| Colonne | Description |
|---|---|
| `id_compte` | Identifiant du compte |
| `type_client` | `PARTICULIER`, `PROFESSIONNEL` ou `ENTREPRISE` |
| `region` | Région du compte (8 régions) |
| `univers` | Offre principale : `MOBILE`, `FIBRE`, `BOX_4G`, `FIXE` |
| `nb_lignes` | Nombre de lignes du compte |
| `mode_paiement` | Mode de paiement habituel : `MOBILE_MONEY`, `ESPECES`, `PRELEVEMENT`, `CARTE`, `VIREMENT` |
| `canal_facture` | Réception de la facture : `SMS`, `EMAIL`, `PAPIER` |
| `contact_valide` | 1 si le contact de facturation est valide (numéro ou e-mail joignable) |
| `date_ouverture` | Date d'ouverture du compte |

## `data/factures.csv` — ≈ 100 000 factures mensuelles (janvier 2024 à septembre 2026)
| Colonne | Description |
|---|---|
| `id_facture`, `id_compte` | Identifiants |
| `periode` | Mois facturé `AAAA-MM` |
| `date_emission` | Émission : le 3 du mois suivant la période |
| `date_echeance` | Échéance : émission + 15 jours |
| `montant` | Montant en FCFA |
| `date_solde` | Date à laquelle la facture a été entièrement payée ; vide si elle ne l'est pas au 31/10/2026 |
| `delai_paiement_jours` | `date_solde − date_echeance` (négatif = payée en avance) ; vide si non soldée |
| `risque_impaye` | **Cible** : 1 si non soldée 60 jours après l'échéance, 0 sinon ; vide si l'échéance + 60 jours dépasse le 31/10/2026 |

## `data/dataset_modelisation.csv` — table d'apprentissage (une ligne par facture)
Toutes les variables d'historique sont calculées **à la date d'émission**, avec ce qu'on savait ce jour-là.

| Variable | Description |
|---|---|
| `type_client`, `region`, `univers`, `mode_paiement`, `canal_facture`, `nb_lignes`, `contact_valide` | Caractéristiques du compte |
| `anciennete_mois` | Ancienneté du compte à l'émission |
| `mois_emission` | Mois de l'émission (saisonnalité) |
| `montant` | Montant de la facture |
| `ratio_montant_6m` | Montant / moyenne des 6 factures précédentes (repère les factures anormalement élevées) |
| `nb_factures_passees` | Nombre de factures précédentes |
| `retard_moyen_6m`, `retard_max_6m` | Retard moyen et pire retard sur les 6 factures précédentes (retard courant si encore impayée) |
| `retard_derniere_facture` | Retard de la facture précédente au jour de l'émission |
| `derniere_facture_payee` | 1 si la facture précédente était déjà payée au jour de l'émission |
| `tendance_retard` | Retard moyen des 3 dernières factures − celui des 3 précédentes |
| `part_a_temps_12m` | Part des 12 dernières factures payées au plus tard à l'échéance |
| `retard_habituel_12m` | Médiane des délais des factures payées sur 12 mois (le « rythme » du payeur) |
| `jours_depuis_dernier_paiement` | Jours écoulés depuis le dernier paiement |
| `nb_impayes_en_cours`, `montant_impaye_en_cours` | Factures précédentes non payées au jour de l'émission, et leur montant |
| `nb_impayes_60j_12m` | Impayés à 60 jours déjà constatés sur les 12 dernières factures |
| `delai_paiement_jours`, `risque_impaye` | Issue (ne sert jamais de variable explicative) |

## `data/verite_cachee.csv`
Vraie probabilité d'impayé utilisée par le simulateur. Elle n'existe que parce que les données sont simulées et
sert **uniquement** à mesurer le plafond atteignable (AUC « oracle ») ; elle n'entre jamais dans l'apprentissage.
