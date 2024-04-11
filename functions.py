#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Librairies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import kruskal


# In[2]:


def display_dataset_dimensions(df):
    """
    Affiche les dimensions d'un DataFrame, y compris le nombre d'observations (ou articles) et de colonnes.

    Parameters:
    df (pandas.DataFrame): Le DataFrame dont les dimensions doivent être affichées.

    Returns:
    None. Affiche les dimensions d'un DataFrame.
    """
    print("Le tableau comporte {} observation(s) ou article(s).".format(df.shape[0]))
    print("Le tableau comporte {} colonne(s).".format(df.shape[1]))


# In[3]:


def display_missing_values_percentage(df):
    """
    Affiche les colonnes d'un DataFrame avec le pourcentage de valeurs manquantes,
    triées par ordre croissant de ce pourcentage.

    Parameters:
    df (pandas.DataFrame): Le DataFrame à analyser.

    Returns:
    None. Affiche le pourcentage de valeurs manquantes pour chaque colonne.
    """
    # Calcul du pourcentage de valeurs manquantes pour chaque colonne
    missing_percentage = df.isnull().mean() * 100
    
    # Tri du pourcentage de valeurs manquantes par ordre croissant
    missing_percentage_sorted = missing_percentage.sort_values()
    
    # Affichage des résultats
    print("Pourcentage de valeurs manquantes par colonne :")
    print(missing_percentage_sorted)


# In[4]:


def remove_columns_with_missing_values(df, threshold_percentage):
    """
    Supprime les colonnes d'un DataFrame qui ont plus de 'threshold_percentage' % de valeurs manquantes.

    Parameters:
    df (pandas.DataFrame): Le DataFrame à traiter.
    threshold_percentage (float): Le seuil de pourcentage de valeurs manquantes pour supprimer une colonne.

    Returns:
    pandas.DataFrame: Un DataFrame avec les colonnes supprimées selon le seuil spécifié.
    """
    # Calcul du pourcentage de valeurs manquantes pour chaque colonne
    missing_percentage = df.isnull().mean() * 100
    # Identification des colonnes à supprimer
    columns_to_drop = missing_percentage[missing_percentage > threshold_percentage].index
    # Suppression des colonnes
    df_cleaned = df.drop(columns=columns_to_drop)
    
    return df_cleaned


# In[5]:


def remove_rows_with_all_100g_columns_empty(df):
    """
    Supprime les lignes pour lesquelles toutes les colonnes se terminant par "_100g" sont nulles.

    Parameters:
    df (pandas.DataFrame): Le DataFrame à traiter.

    Returns:
    pandas.DataFrame: Un DataFrame avec les lignes supprimées où toutes les colonnes se terminant par "_100g" sont nulles.
    """
    # Identifier les colonnes qui se terminent par "_100g"
    columns_100g = [col for col in df.columns if col.endswith('_100g')]
    
    # Supprimer les lignes où toutes ces colonnes sont nulles
    df_cleaned = df.dropna(subset=columns_100g, how='all')
    
    return df_cleaned


# In[6]:


def remove_rows_where_column_is_null(df, column_name):
    """
    Supprime les lignes d'un DataFrame où la valeur dans une colonne spécifiée est nulle.

    Parameters:
    df (pandas.DataFrame): Le DataFrame à traiter.
    column_name (str): Le nom de la colonne sur laquelle vérifier les valeurs nulles.

    Returns:
    pandas.DataFrame: Un DataFrame avec les lignes supprimées où la valeur dans la colonne spécifiée est nulle.
    """

    # Suppression des lignes où la valeur dans la colonne spécifiée est nulle
    cleaned_df = df.dropna(subset=[column_name])
    
    return cleaned_df


# In[7]:


def separate_quantitative_and_qualitative_columns(df):
    """
    Sépare les noms des colonnes quantitatives et qualitatives d'un DataFrame.

    Parameters:
    df (pandas.DataFrame): Le DataFrame à analyser.

    Returns:
    tuple: Un tuple contenant deux listes, la première avec les noms des colonnes quantitatives et la seconde avec les noms des colonnes qualitatives.
    """
    # Sélection des noms des colonnes quantitatives (numériques)
    quantitative_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Sélection des noms des colonnes qualitatives (catégorielles)
    qualitative_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    
    return quantitative_columns, qualitative_columns


# In[8]:


def count_unique_values(df, columns):
    """
    Compte le nombre de valeurs uniques pour chaque colonne spécifiée dans un DataFrame.

    Parameters:
    df (pandas.DataFrame): Le DataFrame à analyser.
    columns (list): Liste des noms des colonnes pour lesquelles compter les valeurs uniques.

    Returns:
    dict: Un dictionnaire avec les noms des colonnes comme clés et le nombre de valeurs uniques comme valeurs.
    """
    unique_counts = {}
    for column in columns:
        unique_counts[column] = df[column].nunique()
        
    return unique_counts


# In[9]:


def plot_pie_of_column(df, column_name, title_pieplot):
    """
    Affiche un pie plot des valeurs d'une colonne spécifiée dans un DataFrame.

    Parameters:
    df (pandas.DataFrame): Le DataFrame contenant les données.
    column_name (str): Le nom de la colonne pour laquelle afficher le pie plot.
    title_pieplot (str): Le titre du pie plot.

    Returns:
    None. Affiche un diagramme en secteurs des valeurs de la colonne.
    """
    # Calcul du décompte des valeurs uniques
    value_counts = df[column_name].value_counts()
        
    # Création du pie plot
    plt.figure(figsize=(8, 8))  # Taille du plot
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'{title_pieplot}')
    plt.axis('equal')  # Assure que le pie plot est circulaire.
    plt.show()


# In[10]:


def replace_values_outside_range(df, columns, min_val, max_val):
    """
    Remplace par NaN les valeurs des colonnes spécifiées qui sont strictement inférieures à min_val
    ou strictement supérieures à max_val.

    Parameters:
    df (pandas.DataFrame): Le DataFrame à modifier.
    columns (list): Liste des noms des colonnes à vérifier.
    min_val (int/float): La valeur minimale autorisée.
    max_val (int/float): La valeur maximale autorisée.

    Returns:
    pandas.DataFrame: Le DataFrame avec les valeurs remplacées par NaN si elles sont hors des limites spécifiées.
    """
    df_copy = df.copy()
    for column in columns:
        # Utilisation de apply pour remplacer les valeurs hors de l'intervalle [min_val, max_val] par NaN
        df_copy[column] = df[column].apply(lambda x: np.nan if x < min_val or x > max_val else x)
        
    return df_copy


# In[11]:


def replace_outliers_with_nan(df, columns):
    """
    Remplace les valeurs aberrantes dans les colonnes spécifiées d'un DataFrame par NaN,
    en utilisant la méthode de l'écart interquartile (IQR).

    Parameters:
    df (pandas.DataFrame): Le DataFrame à nettoyer.
    columns (list): Liste des noms des colonnes dans lesquelles chercher et remplacer les valeurs aberrantes.

    Returns:
    pandas.DataFrame: Un DataFrame avec les valeurs aberrantes remplacées par NaN dans les colonnes spécifiées.
    """
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remplacer les valeurs aberrantes par NaN
        df[column] = df[column].apply(lambda x: np.nan if x < lower_bound or x > upper_bound else x)
    
    return df


# In[12]:


def plot_columns(df, columns, plot_type):
    """
    Affiche des diagrammes spécifiés pour chaque colonne donnée d'un DataFrame.

    Parameters:
    df (pandas.DataFrame): Le DataFrame à utiliser.
    columns (list): Liste des noms des colonnes pour lesquelles afficher les diagrammes.
    plot_type (str): Type de diagramme à afficher ("scatter plot", "box plot", "histplot").

    Returns:
    None. Affiche les diagrammes demandés.
    """
    # Définir le nombre de lignes nécessaires en fonction du nombre de colonnes
    nrows = len(columns) // 2 + (len(columns) % 2 > 0)
    
    # Initialiser la figure matplotlib
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 4 * nrows))
    axs = axs.flatten()  # Aplatir le tableau d'axes pour un accès plus facile

    for i, column in enumerate(columns):
        ax = axs[i]
        if plot_type == "boxplot":
            sns.boxplot(x=column, data=df, ax=ax)
        elif plot_type == "histplot":
            sns.histplot(df[column], kde=True, ax=ax)
        else:
            print(f"Type de diagramme '{plot_type}' non reconnu.")
            return
        
        ax.set_title(f'{plot_type} de {column}')
    
    # Ajuster le layout pour éviter le chevauchement
    plt.tight_layout()
    plt.show()


# In[ ]:


def impute_nutrition_grade_with_knn(df, numeric_columns):
    """
    Impute les valeurs manquantes de 'nutrition_grade_fr' dans un DataFrame en utilisant KNN.
    
    Parameters:
    df (pandas.DataFrame): Le DataFrame à traiter.
    numeric_columns (list): Liste des colonnes numériques à utiliser pour l'imputation.
    
    Returns:
    pandas.DataFrame: DataFrame avec les valeurs de 'nutrition_grade_fr' imputées.
    """
    df_copy = df.copy()
    
    # Prétraitement : Normalisation des données numériques
    scaler = StandardScaler()
    df_copy[numeric_columns] = scaler.fit_transform(df_copy[numeric_columns])
    
    # Séparation des données
    known_grade = df_copy[df_copy['nutrition_grade_fr'].notna()]
    unknown_grade = df_copy[df_copy['nutrition_grade_fr'].isna()]
    
    # Encodeur pour les variables catégorielles
    le = LabelEncoder()
    X_known = known_grade[numeric_columns]
    y_known = le.fit_transform(known_grade['nutrition_grade_fr'])
    
    # Configuration de GridSearchCV pour trouver le meilleur 'n_neighbors'
    param_grid = {'n_neighbors': np.arange(1, 10)}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_known, y_known)
    
    # Meilleur 'n_neighbors' et score
    best_k = grid_search.best_params_['n_neighbors']
    best_score = grid_search.best_score_
    print(f"Le nombre optimal de voisins est : {best_k}")
    print(f"Meilleure score d'exactitude avec k={best_k} : {best_score}")
    
    # Entraînement et prédiction avec le meilleur 'k'
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_known, y_known)
    X_unknown = unknown_grade[numeric_columns]
    predicted_grade = knn.predict(X_unknown)
    predicted_grade_labels = le.inverse_transform(predicted_grade)
    
    # Remplissage des valeurs manquantes dans le DataFrame original
    df_copy.loc[df_copy['nutrition_grade_fr'].isna(), 'nutrition_grade_fr'] = predicted_grade_labels
    
    return df_copy


# In[ ]:


def impute_with_mean_by_group(df, target_col, group_col):
    """
    Impute les valeurs manquantes dans la colonne cible d'un DataFrame par la moyenne de cette colonne,
    groupée par une autre colonne spécifiée.
    
    Parameters:
    df (pandas.DataFrame): Le DataFrame à traiter.
    numeric_columns (list): Liste des colonnes numériques à utiliser pour l'imputation.
    
    Returns:
    pandas.DataFrame: DataFrame avec les valeurs de 'nutrition_grade_fr' imputées.
    """
    # Calcul des moyennes par groupe pour la colonne cible
    mean_by_group = df.groupby(group_col)[target_col].mean()

    # Assure que les groupes de group_col qui sont NaN ne sont pas utilisés pour l'imputation
    valid_groups = df[group_col].notna()

    # Imputation des valeurs manquantes dans la colonne cible pour les groupes valides
    for group, mean in mean_by_group.iteritems():
        df.loc[(df[group_col] == group) & (df[target_col].isna()), target_col] = mean
    
    return df


# In[ ]:


def perform_pca_and_return_pca_object(df, quantitative_columns):
    """
    Effectue une ACP sur un DataFrame en utilisant des colonnes quantitatives spécifiées,
    affiche l'éboulis des valeurs propres, détermine le nombre de composantes à retenir,
    et retourne l'objet PCA.

    Parameters:
    df (pandas.DataFrame): Le DataFrame à traiter.
    quantitative_columns (list): Liste des colonnes quantitatives à utiliser pour l'ACP.

    Returns:
    PCA: L'objet PCA après avoir été ajusté aux données.
    data_scaled : Les données scalées.
    """
    # Scaler les données quantitatives
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[quantitative_columns])
    
    # Effectuer l'ACP
    pca = PCA()
    pca.fit(data_scaled)
    
    # Afficher l'éboulis des valeurs propres
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.title("Éboulis des valeurs propres")
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance expliquée cumulée")
    plt.axhline(y=0.8, color='r', linestyle='-')
    plt.text(0.5, 0.85, '80% variance expliquée', color = 'red', verticalalignment='center')
    plt.show()
    
    # Déterminer le nombre de composantes à retenir
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    nb_components = np.argmax(cumulative_variance >= 0.8) + 1  # Ajoute 1 car les indices commencent à 0
    print(f"Nombre de composantes à analyser pour atteindre au moins 80% de variance expliquée: {nb_components}")
    
    return pca, data_scaled


# In[ ]:


def plot_pca_results_colored(pca, pca_scores, quantitative_columns, qualitative_column):
    """
    Affiche les projections des individus sur F1/F2 et F3/F4 (si possible), ainsi que les cercles de corrélation
    pour les deux premiers axes principaux avec des couleurs correspondant à la variable qualitative.

    Parameters:
    pca : Objet PCA après ajustement aux données.
    pca_scores : Scores PCA des individus sur les composantes principales.
    quantitative_columns : Liste des variables quantitatives.
    qualitative_column : Données de la variable qualitative pour la coloration des points.
    """
    # Détermination du nombre de composantes pour les graphiques
    pairs_of_components = [(0, 1), (2, 3)]

    # Création de la figure avec des sous-graphiques
    fig, axes = plt.subplots(len(pairs_of_components), 2, figsize=(30, 30))

    for i, (component_a, component_b) in enumerate(pairs_of_components):
        ax_individuals, ax_correlation = axes[i]
        
        # Projection des individus avec couleurs basées sur la variable qualitative
        sns.scatterplot(x=pca_scores[:, component_a], y=pca_scores[:, component_b],
                        hue=qualitative_column, palette='Set1', alpha=0.7, ax=ax_individuals)
        ax_individuals.set_title(f"Projection des individus sur 'F{component_a + 1}' et 'F{component_b + 1}'")
        ax_individuals.set_xlabel(f"F{component_a + 1} ({pca.explained_variance_ratio_[component_a]:.2%} variance)")
        ax_individuals.set_ylabel(f"F{component_b + 1} ({pca.explained_variance_ratio_[component_b]:.2%} variance)")
        ax_individuals.legend(title=qualitative_column.name, bbox_to_anchor=(1, 1), loc='upper left')
        ax_individuals.grid(True)

        # Cercle de corrélation
        for j, (x_variable, y_variable) in enumerate(zip(pca.components_[component_a], pca.components_[component_b])):
            ax_correlation.arrow(0, 0, x_variable, y_variable, head_width=0.05, head_length=0.1, fc='k', ec='k')
            ax_correlation.text(x_variable + 0.05, y_variable + 0.05, quantitative_columns[j], fontsize=20)

        ax_correlation.set_title(f"Cercle des corrélations (F{component_a + 1} et F{component_b + 1})")
        ax_correlation.set_xlabel(f"F{component_a + 1}")
        ax_correlation.set_ylabel(f"F{component_b + 1}")
        ax_correlation.axhline(0, color='grey', linestyle='--')
        ax_correlation.axvline(0, color='grey', linestyle='--')
        ax_correlation.set_xlim(-1, 1)
        ax_correlation.set_ylim(-1, 1)
        circle = plt.Circle((0, 0), 1, edgecolor='blue', facecolor='none', linestyle='-', linewidth=2)
        ax_correlation.add_artist(circle)
        ax_correlation.grid(True)

    plt.tight_layout()
    plt.show()


# In[ ]:


def kruskal_wallis_test(df, quantitative_columns, qualitative_column):
    """
    Réalise le test non paramétrique de Kruskal-Wallis sur un DataFrame. Affiche les résultats sous forme 
    d'un tableau et génère des boxplots pour chaque variable quantitative.

    Parameters:
    df (pd.DataFrame): DataFrame contenant les données.
    quantitative_columns (list): Liste des colonnes quantitatives à tester.
    qualitative_column (str): Colonne qualitative utilisée pour grouper les données dans le test et les boxplots.
    """
    # Création d'un tableau pour les résultats
    results = pd.DataFrame(columns=['Variable', 'H-value', 'P-value', 'Eta-squared'])

    # Réalisation du test de Kruskal-Wallis pour chaque variable quantitative
    for col in quantitative_columns:
        groups = [group.dropna().values for name, group in df.groupby(qualitative_column)[col]]
        H, p = kruskal(*groups)
        eta_squared = np.var([np.mean(group) for group in groups]) / np.var(df[col].dropna())
        results = results.append({'Variable': col, 'H-value': H, 'P-value': p, 'Eta-squared': eta_squared}, ignore_index=True)
    
    # Affichage du tableau des résultats
    print(results)

    # Tri de la colonne qualitative par ordre croissant pour l'affichage
    sorted_qualitative_values = sorted(df[qualitative_column].unique())
    
    # Affichage des boxplots sans les outliers et avec la médiane marquée
    nbr_rows = int(np.ceil(len(quantitative_columns) / 2))
    fig, axs = plt.subplots(nbr_rows, 2, figsize=(14, nbr_rows * 5))

    for i, col in enumerate(quantitative_columns):
        row = i // 2
        col_num = i % 2
        ax = axs[row, col_num] if nbr_rows > 1 else axs[col_num]
        
        # Boxplot avec tri des valeurs qualitatives
        sns.boxplot(x=qualitative_column, y=col, data=df, showfliers=False, ax=ax, order=sorted_qualitative_values,
                    palette="Set1", medianprops={'color': 'black'})
        
        # Calcul de la médiane et ajout sur le boxplot
        medians = df.groupby(qualitative_column)[col].median().reindex(sorted_qualitative_values)
        for xtick in ax.get_xticks():
            ax.plot(xtick, medians[sorted_qualitative_values[xtick]], 'ko')
        
        ax.set_title(f'Boxplot de {col} par {qualitative_column}')

    plt.tight_layout()
    plt.show()

