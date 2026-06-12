"""WebApp Streamlit : risque de départ d'un employé.

Lancement :
    pip install streamlit
    streamlit run app.py

On n'expose que quelques mesures clés à l'utilisateur RH ; les autres features sont
remplies avec une valeur moyenne du jeu d'entraînement.

Auteur de cette partie : Chaimae IMRANI
"""
import numpy as np
import joblib
import streamlit as st

BUNDLE = joblib.load("modele.joblib")
FEATURES = BUNDLE["features"]
PLAGES = BUNDLE["plages"]

# Valeur par défaut de chaque feature = milieu de la plage d'entraînement
DEFAUTS = (PLAGES["min"] + PLAGES["max"]) / 2

st.title("Risque de départ d'un employé")
st.write("Renseignez quelques infos ; le modèle estime la probabilité que l'employé quitte l'entreprise.")

valeurs = DEFAUTS.copy()


def champ(nom, label, **kw):
    """Affiche un input pour une feature si elle existe, et met à jour le vecteur."""
    if nom in FEATURES:
        i = FEATURES.index(nom)
        valeurs[i] = st.number_input(label, value=float(round(DEFAUTS[i])), **kw)


champ("Age", "Âge", min_value=18.0, max_value=65.0)
champ("MonthlyIncome", "Salaire mensuel", min_value=1000.0, max_value=20000.0)
champ("TotalWorkingYears", "Années d'expérience totale", min_value=0.0, max_value=40.0)
champ("YearsAtCompany", "Ancienneté dans l'entreprise", min_value=0.0, max_value=40.0)
champ("JobSatisfaction", "Satisfaction au travail (1 à 4)", min_value=1.0, max_value=4.0)
overtime = st.checkbox("Fait des heures supplémentaires")
if "OverTime" in FEATURES:
    valeurs[FEATURES.index("OverTime")] = 1.0 if overtime else 0.0

if st.button("Prédire"):
    x = valeurs.reshape(1, -1)
    x_s = BUNDLE["scaler"].transform(x)
    pred = int(BUNDLE["modele"].predict(x_s)[0])
    proba = float(BUNDLE["modele"].predict_proba(x_s)[0][1])

    if pred == 1:
        st.error(f"⚠️ Risque de départ (probabilité {proba:.0%})")
    else:
        st.success(f"✅ Va probablement rester (probabilité de départ {proba:.0%})")
    st.progress(proba)
    st.caption("À combiner avec le jugement du manager : le modèle aide, il ne décide pas.")
