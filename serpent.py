import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import joblib

# titre du site
st.set_page_config(layout='wide', page_icon="https://github.com/mogdan/Datascientest_CO2/blob/main/streamlit_assets/Green_co2_logo2.png?raw=true")
col1, col2, col3 = st.columns([1, 10, 1])
col2.title(":green[Serpent]")

# menu gauche de navigation
st.sidebar.image("assets/snake_cute.jpg", use_column_width=True)
st.sidebar.title("Sommaire")
# accès aux pages du site
pages=["1 - Partie 1 ", "2 - Partie 2", "3 - Partie 3"]
page=st.sidebar.radio("Aller vers la page :", pages)

# contenu des pages sélectionnées
if page == pages[0]: 
  st.header('1 - Exploration des datasets', divider=True)
  st.markdown("# :grey[Prise en main du sujet]")

  with st.expander("Problématique"):
    problematique = '''
    :green[L’**accumulation de gaz à effet de serre**] dans l’atmosphère est l’une des principales causes de réchauffement climatique. 
    
    Or, les transports, et principalement la voiture, sont la première source de gaz à effet de serre en France.
    Selon l'ADEME, les Français affectionne particulièrement ce mode de transport à **77%** malgré l'attractivité des autres modes de transports.

    En tant que Français, La voiture est donc responsable d’une part importante de notre empreinte carbone au quotidienn. 
    
    Face à l’urgence climatique, certains constructeurs ont déjà œuvré ces dix dernières années sur :
    - l’amélioration des rendements des moteurs thermiques
    - l’aérodynamisme
    - l’allègement des voitures 

    Intéressés par cette problématique, nous avons ainsi choisi ce sujet pour mettre à profit nos nouvelles compétences en tant que Data Analyst. 

    Les objectifs pour notre équipe sur ce sujet sont les suivants :
    -	:green[**Consolider les données d’étude**] : Rechercher, analyser et nettoyer les données à notre disposition sur ce sujet
    -	:green[**Concevoir un modèle de prédiction**] :  pour déterminer les émissions de CO2 en fonction des caractéristiques des véhicules
  
    '''
    st.markdown(problematique)

  with st.expander("Sources des Données"):
      ademe = '''
      **[Source ADEME](https://www.data.gouv.fr/fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france/)**  
      - Données françaises
      - Données allant de 2002 jusqu’à 2015
      - 300K données
      '''
      st.markdown(ademe)

      ue = '''
      **[Source UE](https://www.eea.europa.ea/data-and-maps/data/co2-cars-emission-20)**  
      - Données européennes (dont données françaises)
      - Données allant de 2010 à 2022
      - 80M données
      '''
      st.markdown(ue)

  st.markdown("# :grey[Exploration des données]")

  with st.expander("Données ADEME"):      
        st.markdown("L‘illustration suivante permet de se rendre compte de la qualité macro des données ADEME :")
        st.image("https://github.com/mogdan/Datascientest_CO2/blob/main/streamlit_assets/ademe_raw.png?raw=true", use_column_width=True)

        ademe_choice = '''
        Pour chaque fichier est indiqué :
        -	Le nb de lignes
        -	Le nb / titres des colonnes

        Notre analyse est alors la suivante :
        -	**Niveau cohérence** : disparités fortes au niveau du format des données
        -	**Niveau complétude** : pour certaines années il manque de la donnée (2004). Il y a également des très fortes disparités du nombre d’entrées entre les années

        '''
        st.markdown(ademe_choice)

  with st.expander("Données UE"):      
        st.markdown("Au niveau des données UE, nous constatons rapidement que nous sommes sur un set de données déjà standardisées sur le périmètre européen de 2010 à 2023. ")
        st.image("https://github.com/mogdan/Datascientest_CO2/blob/main/streamlit_assets/ue_raw.png?raw=true", use_column_width="auto")
        
        ue_choice = '''
        Notre analyse est alors la suivante :
        -	**Niveau cohérence** : les données sont déjà standardisées dans un même format
        -	**Niveau complétude** : il y a également sur ce set de données des écarts importants sur les enregistrements entre certaines années
        -	**Niveau actualité** : ces données semblent être à jour
        '''
        st.markdown(ue_choice)

  with st.expander("Variables"):      
        st.markdown("Pour commencer le travail exploratoire, nous avons décidé de faire un heatmap pour regarder les corrélations entre les valeurs numériques. Pou_r se faire, nous avons choisi de remplacer les NaN par la moyenne dans chaque variable.")
        st.image("https://github.com/mogdan/Datascientest_CO2/blob/main/streamlit_assets/Heatmap.png?raw=true", use_column_width="auto")
        
        corr_model = '''
        Sur ce graphique, 3 corrélations supérieures à 80% peuvent être observées :
        -	**Mt avec m (kg)** : ces 2 variables représentent le poids des véhicules. Il est donc logique qu’elles soient corrélées.
        -	**Enedc (g/km) avec Ewltp (g/km)** : Ces 2 variables représentent les émissions de CO2 calculées avec des normes différentes. Il y a forte corrélation entre les deux ce qu’il s’avère plutôt logique.
        -	**M (kg) avec At2 (mm)** : La variable At2 représente la distance entre les 2 roues AV ou AR d’un véhicule, tout comme l’At1 d’ailleurs. Après une rapide analyse, on remarque que la variable At2 avait plus de NaN que sa consœur et que notre remplacement par la moyenne ait pu avoir des effets de bord.
        '''
        st.markdown(corr_model)

  with st.expander("Critères ENEDC / EWLTP"):  
        critere = '''
        Ce sont 2 indicateurs de mesure des émissions de CO2 : 
        -	**NEDC** signifiant « New European Driving Cycle » ou « Nouveau Cycle de Conduite Européen », c’est une norme d’homologation des véhicules neufs introduite en 1997 en Europe et qui a eu cours jusqu'en septembre 2017. La norme va définir les conditions dans lesquelles un modèle est testé, allant de la vitesse à la température, avec un avantage : tous les véhicules suivent le même protocole.
        -	**WLTP** (Worldwide Harmonized Light Vehicles Test Procedure) pour tout nouveau modèle à partir du 1er septembre 2017. Il concerne tous les véhicules neufs au 1er septembre 2018, et jusqu’aux véhicules en stock homologués NEDC et vendus après le 1er septembre 2019.
        '''
        st.markdown(critere)
        st.image("https://github.com/mogdan/Datascientest_CO2/blob/main/streamlit_assets/Comparaison ENEDC-EWLTP par an.png?raw=true", use_column_width="auto")
        
        critere1 = '''
        Grâce à cette analyse, c'est à cette étape que nous avons choisi d'exclure les années 2015 et 2016 dans la suite de notre étude à cause du faible nombre de données.
        
        :green[Sur les autres années, notre analyse a été la suivante :]
        -	En 2017 et 2018, nous avons des données ENEDC et peu voire pas de données EWLTP
        -	Durant la période 2019 – 2020, il y a coexistence des données sur ces 2 types de mesures.
        -	En revanche cette tendance qui s’inverse à partir 2021 où les données ENEDC sont majoritairement absentes.

        Pour pousser notre analyse d'un cran supplémentaire, nous nous sommes concentrés sur l'année 2020 où nous avons des données complètes sur les 2 critères. 
        '''

        st.markdown(critere1)
        st.image("https://github.com/mogdan/Datascientest_CO2/blob/main/streamlit_assets/Boxplot - comparasion 2020 ENEDC-EWLTP.png?raw=true", use_column_width="auto")
        critere2 = '''
        Ce que nous avons voulu mettre en évidence ici est la médiane des émissions par type de carburant et par norme (Enedc puis Ewltp), et la différence par type de carburant.

        :green[**Conclusion** : 
        - Grâce à ce graphique, npous pouvons confirmer que les émissions calculées avec la norme NEDC sont plus faibles qu’avec la norme WLTP.
        - Nous pouvons tenter de reconstruire des donnes proches des données EWLTP sur les années où ce référentiel n'était pas en application]

        '''
        st.markdown(critere2)

  with st.expander("Approfondissements"):  
        distrib = '''
        Pour continuer notre travail exploratoire, nous avons regardé le nombre de véhicules par type de carburant sur ce jeu de données.
        '''
        st.markdown(distrib)
        st.image("https://github.com/mogdan/Datascientest_CO2/blob/main/streamlit_assets/Nb véhicules par type de carburant.png?raw=true", use_column_width="auto")
        
        distrib1 = '''
        :green[**Nous observons ainsi une grande prédominance des voitures Essence et Diesel dans nos données.**]
        
        Pour compléter ce graphique, il nous a semblé intéressant d’observer **l’évolution dans le temps pour chaque carburant** :
        '''
        st.markdown(distrib1)
        st.image("https://github.com/mogdan/Datascientest_CO2/blob/main/streamlit_assets/Nb véhicules par type de carburant et par an.png?raw=true", use_column_width="auto")
        distrib2 = '''
        
        :green[**Nous observons une diminution progressive des immatriculations pour les voitures Essence / Diesel** ces dernières années et une augmentation progressive des voitures électriques à partir de 2019. 
        Cependant, ces graphiques ne permettent pas d’observer de causes conjoncturelles ou structurelles pour ces tendances.]

        Pour aller plus loin, on s'est intéressé à la distribution par type de carburant et la taille de cylindrée pour voir comment sont répartis les véhicules.
        
        '''
        st.markdown(distrib2)
        st.image("https://github.com/mogdan/Datascientest_CO2/blob/main/streamlit_assets/Distribution par type de carburant et taille de cylindrée.png?raw=true", use_column_width="auto")
        
        distrib3 = '''
        Nous pouvons voir pour les catégories qui nous intéressent :
        -	**Essence :** Les véhicules essences sont en général bien équilibrées dans leur répartition, la moyenne se situant quasi au même niveau que la médiane. Il y a toutefois une large dispersion de VA (grosses cylindrées – ex. sport, luxe)
        -	**Diesel :** la cylindrée est en général plus importante que pour l’homologue en version essence et possède une dispersion plus faible de valeurs aberrantes.
        '''
        st.markdown(distrib3)


  st.markdown("# :grey[Conclusion de l'Analyse Exploratoire]")

  with st.expander("Choix des données"):      
        st.markdown("Au vu des problématiques vues précédemment sur les données ADEME, nous avons décidé de nous concentrer sur les :green[**données UE.**] ")
  
  with st.expander("Choix de la période d'observation"):      
        period_choice = '''
        :green[**Cette étape a pour enjeu de nous permettre de sélectionner la plage de données sur laquelle va s’appuyer notre modèle.**]
        
        Pour notre étude, nous avons choisi la période « 2017 – 2022 ».
        -	Avant 2017, nous n'avions pas beaucoup de données
        -	Le faible volume de données 2023 traduit des données non encore intégrées (année en cours)
        '''
        st.markdown(period_choice)

elif page == pages[1]:
  st.header('Partie 2', divider=True)

elif page == pages[2]:
  st.header('Partie 3', divider=True)
