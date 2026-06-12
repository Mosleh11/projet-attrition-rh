# Projet de fin de module — Prédiction de l'attrition des employés

**Groupe :** Mohammed MOSLEH & Chaimae IMRANI
Cours *Introduction au ML / Deep Learning* — IPSSI Mastère Dev-Data-IA 4ᵉ année.

## Le problème

À partir des données RH d'une entreprise (âge, salaire, ancienneté, poste, satisfaction,
heures supplémentaires…), prédire si un employé risque de **quitter l'entreprise**
(`Attrition` : Yes/No). C'est de la **classification binaire supervisée**.

**Objectif métier :** aider le service RH à repérer les employés à risque de départ pour
cibler les actions de rétention, au lieu d'agir à l'aveugle.

## Le dataset

IBM HR Analytics Employee Attrition & Performance — 1470 employés, 35 colonnes.
🔗 https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
Fichier : `data/HR-Employee-Attrition.csv`. Cible déséquilibrée : ~16 % de départs.

## Organisation du projet (qui a fait quoi)

| Partie | Auteur | Fichier | Contenu |
|--------|--------|---------|---------|
| 1 — Données | **Mohammed MOSLEH** | `01_exploration_preprocessing.ipynb` | Audit, exploration, nettoyage, encodage, multicolinéarité (VIF), sélection de features → produit `data/attrition_clean.csv` |
| 2 — Modèle | **Chaimae IMRANI** | `02_modelisation_evaluation.ipynb` | Split 3 jeux, Arène des algos, validation croisée, coût métier, champion, sérialisation |
| 2 — Déploiement | **Chaimae IMRANI** | `api.py`, `app.py` | API Flask + WebApp Streamlit |

Les deux notebooks s'enchaînent : la Partie 1 sort un CSV propre que la Partie 2 consomme.

## Résultats

**L'Arène** (sur du déséquilibré, on juge au F1 / recall, pas à l'accuracy) :

| Algo | Accuracy | Recall (partants) | F1 |
|------|----------|-------------------|----|
| SVC rbf | 0.85 | 0.55 | 0.54 |
| Gradient Boosting | 0.87 | 0.38 | 0.49 |
| **Régression logistique** | 0.76 | **0.70** | 0.48 |
| Random Forest | 0.86 | 0.17 | 0.28 |
| KNN | 0.86 | 0.13 | 0.22 |

**Coût métier** (rater un partant = 10, fausse alerte = 1) :
la régression logistique a le coût le plus bas (**198**) contre 299 (GB), 393 (RF) et
**470** pour la baseline « personne ne part ».

## Le champion : la régression logistique

On déploie la **régression logistique** (avec `class_weight="balanced"`) :
- **meilleur recall sur les partants (0.70)** — l'essentiel ici : rater un départ coûte cher ;
- **coût métier le plus bas** ;
- **rapide et explicable** : on peut dire au RH *pourquoi* un employé est à risque
  (heures supplémentaires, salaire, ancienneté…), ce qui compte en RH.

Verdict test : AUC = 0.78, recall partants = 0.66.

> Sur ce dataset tabulaire et déséquilibré, l'accuracy seule ment (la baseline fait 84 %
> sans rien comprendre). C'est le recall et le coût métier qui désignent le vrai champion.

## Lancer le projet

```bash
pip install -r requirements.txt

# 1. Partie données (génère data/attrition_clean.csv)
jupyter notebook 01_exploration_preprocessing.ipynb

# 2. Partie modèle (génère modele.joblib)
jupyter notebook 02_modelisation_evaluation.ipynb

# 3. Déploiement
python api.py            # API Flask  (pip install flask)
streamlit run app.py     # WebApp     (pip install streamlit)
```
