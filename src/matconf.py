# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SVC Model - Matrices de Confusión
#
# El objetivo de este notebook extraer las matrices de confusión de cada fold del entrenamiento por Cross-validation
#
# ## Integrantes:
#
# * Del Villar, Javier
# * Pistoya, Haydeé Soledad
# * Sorza, Andrés
#
# ## Cargamos Librerías

# %%
import joblib
import numpy as np

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC
from typing import Dict

# %% [markdown]
# ## Carga de Datos

# %%
VECTORS_FILE = '../data/02_processed/vectores.joblib'
TARGETS_FILE = '../data/02_processed/targets.joblib'
FEATURE_NAMES_FILE = '../data/02_processed/features.joblib'

# %%
vectores = joblib.load(VECTORS_FILE)
nombres_targets = joblib.load(TARGETS_FILE)
nombres_features = joblib.load(FEATURE_NAMES_FILE)

vectores = vectores[nombres_targets['mes'] <= '2020-10']

nombres_targets = nombres_targets[nombres_targets['mes'] <= '2020-10'][
    'seccion'
].ravel()

# %% [markdown]
# ## Parámentros

# %%
# Pasar categorias a numeros (1ra categoria = 0, 2da categoria = 1, etc)
label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(nombres_targets)

# idx_a_clase es un diccionario indice de categoria -> nombre de categoria
idx_a_clase = label_encoder.classes_

# Cantidad de categorias distintas que tenemos en el conj. de entrenamiento
n_categorias = len(idx_a_clase)

# El clasificador que vamos a usar
clasificador = SVC(
    C=0.008023390186019818,
    kernel='linear',
    probability=True,
    class_weight='balanced',
    random_state=1234,
)

# Cantidad maxima de features que seleccionara el extractor de features
MAX_FEATURES = 880
# Cantida de folds a usar en cross-val
CANT_FOLDS_CV = 5

# Transformar los targets en N columnas, 1 por cada categoria, donde la categoria correcta tiene
# un 1 y todas las demas columnas en esa fila tienen 0.
# Dado que AUC se calcula sobre 2 categorias, Usamos esto luego para calcular 1 AUC por cada
# categoria
targets_binarios_por_clase = label_binarize(targets, classes=range(0, n_categorias))

# %% [markdown]
# ## Entrenamiento

# %%
# Hacer cross-validation
n_fold = 1
accuracy_promedio = 0

for train_index, test_index in StratifiedKFold(
    n_splits=CANT_FOLDS_CV, random_state=1234, shuffle=True
).split(vectores, targets):
    # Armar folds de entrenamiento para CV
    train_fold = vectores[train_index]
    train_targets_fold = targets[train_index]

    # Armar fold de test para CV
    test_fold = vectores[test_index]
    test_targets_fold = targets[test_index]

    # Seleccionar features a partir de los folds de entrenamiento
    selector_features = SelectKBest(score_func=chi2, k=MAX_FEATURES)
    selector_features.fit(train_fold, train_targets_fold)

    # Dejar en el fold de entrenamiento solo las features seleccionadas con el fold de entrenamiento
    train_fold_selected = selector_features.transform(train_fold)

    # Dejar en el fold de test solo las features seleccionadas con el fold de entrenamiento
    test_fold_selected = selector_features.transform(test_fold)
    selector_features.get_support()

    # Clasificar el fold de test
    preds_fold = clasificador.fit(train_fold_selected, train_targets_fold).predict(
        test_fold_selected
    )
    print(
        f'FOLD #{n_fold}, # instancias train = {train_fold.shape[0]}, # instancias test = {test_fold_selected.shape[0]}\n'
    )

    # Evaluar accuracy comparando las categorias reales con las predichas
    accuracy_fold = accuracy_score(test_targets_fold, preds_fold)
    accuracy_promedio += accuracy_fold
    print(f'Accuracy del fold #{n_fold} = {accuracy_fold}\n')
    print('Matriz de confusion (filas=real, columnas=prediccion):')
    mat_conf = confusion_matrix(test_targets_fold, preds_fold)
    print(mat_conf)
    print('\n')
    n_fold += 1

print(f'Accuracy promedio = {accuracy_promedio / (n_fold - 1)}')
