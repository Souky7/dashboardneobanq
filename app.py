import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Installer streamlit
!pip install -q streamlit

st.title("Score d'éligibilité client")

# Charger le modèle
try:
    model = joblib.load('eligibility_model.joblib')
    st.success("Modèle chargé avec succès.")
except FileNotFoundError:
    st.error("Erreur : Le fichier 'eligibility_model.joblib' n'a pas été trouvé.")
    st.warning("Veuillez vous assurer que le modèle a été entraîné et sauvegardé.")
    model = None

# Charger les données d'entraînement pour les médianes (assurez-vous que ce fichier est disponible)
try:
    df_train = pd.read_csv('application_train.csv')
    df_train = df_train.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    st.success("Données d'entraînement chargées pour les médianes.")
except FileNotFoundError:
    st.error("Erreur : Le fichier 'application_train.csv' n'a pas été trouvé.")
    st.warning("Le graphique des écarts ne pourra pas être généré sans les données d'entraînement.")
    df_train = None


# Uploader le fichier client
uploaded_file = st.file_uploader("Uploader le fichier CSV du client (1 ligne)", type=["csv"])

if uploaded_file is not None:
    client_df = pd.read_csv(uploaded_file)

    st.subheader("Données du client")
    st.dataframe(client_df)

    if model is not None:
        # Supprimer les colonnes non utilisées par le modèle (si présentes)
        client_df_processed = client_df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')

        # Vérifier que les colonnes du client correspondent à celles utilisées lors de l'entraînement
        train_cols = model.named_steps['prep'].feature_names_in_
        client_cols = client_df_processed.columns

        missing_cols = set(train_cols) - set(client_cols)
        if missing_cols:
             st.warning(f"Attention : Colonnes manquantes dans le fichier client par rapport aux données d'entraînement: {', '.join(missing_cols)}")
             st.warning("Le modèle risque de ne pas fonctionner correctement.")

        extra_cols = set(client_cols) - set(train_cols)
        if extra_cols:
            st.warning(f"Attention : Colonnes supplémentaires dans le fichier client par rapport aux données d'entraînement: {', '.join(extra_cols)}")
            client_df_processed = client_df_processed[train_cols]


        # Prédiction
        try:
            proba = model.predict_proba(client_df_processed)[0]
            # proba[0] est la probabilité de la classe 0 (pas de défaut), proba[1] est la probabilité de la classe 1 (défaut)
            eligibility_score = proba[0] # Éligibilité = 1 - proba défaut
            st.subheader("Score d'éligibilité")
            st.write(f"Le score d'éligibilité du client est de : **{eligibility_score:.2f}**")

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")


        # Graphique des écarts
        if df_train is not None:
            st.subheader("Écarts par rapport aux médianes du dataset d'entraînement")

            # Appliquer la même pré-préparation au client pour obtenir les mêmes caractéristiques que l'entraînement
            try:
                client_transformed = model.named_steps['prep'].transform(client_df_processed)
                train_transformed = model.named_steps['prep'].transform(df_train)

                # Obtenir les noms des caractéristiques après transformation
                # Note: Obtenir les noms des caractéristiques transformées du ColumnTransformer peut être complexe.
                # Pour simplifier, on peut se baser sur les noms des colonnes numériques + les noms des OHE.
                # Une méthode plus robuste consisterait à utiliser get_feature_names_out() si disponible et compatible.
                # Pour cet exemple, nous allons nous concentrer sur les caractéristiques numériques transformées.

                # Calculer les médianes des caractéristiques numériques transformées de l'entraînement
                # Identifier les colonnes numériques après transformation
                num_cols_transformed = [f'num__{col}' for col in model.named_steps['prep'].transformers_[1][2]] # Assuming num pipeline is the second one

                if num_cols_transformed:
                    train_num_df = pd.DataFrame(train_transformed, columns=model.named_steps['prep'].get_feature_names_out())[num_cols_transformed]
                    client_num_df = pd.DataFrame(client_transformed, columns=model.named_steps['prep'].get_feature_names_out())[num_cols_transformed]

                    medians_train = train_num_df.median()
                    client_values = client_num_df.iloc[0]

                    # Calculer les écarts
                    diffs = client_values - medians_train

                    # Afficher le graphique des écarts
                    plt.figure(figsize=(10, 6))
                    diffs.sort_values().plot(kind='barh')
                    plt.title('Écart des caractéristiques numériques du client par rapport aux médianes d\'entraînement')
                    plt.xlabel('Écart (Valeur Client - Médiane Entraînement)')
                    plt.ylabel('Caractéristique numérique')
                    st.pyplot(plt)
                else:
                    st.warning("Aucune colonne numérique identifiée après transformation pour le graphique des écarts.")

            except Exception as e:
                st.error(f"Erreur lors de la génération du graphique des écarts : {e}")

    else:
        st.warning("Le modèle n'est pas chargé. Impossible de calculer le score d'éligibilité.")
