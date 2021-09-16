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
# # Baseline Model
#
# El objetivo de este notebook es entrenar un modelo simple para tener una línea base de comparación, para esto se seleccionó un basificador bayesiano multinomial.
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
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.naive_bayes import MultinomialNB

data_from = '../data/02_processed/'
data_to = '../data/03_models/'

walk1_train = ['2021-02', '2021-05']
walk1_test = '2021-06'

walk2_train = ['2021-03', '2021-06']
walk2_test = '2021-07'

walk3_train = ['2021-04', '2021-07']
walk3_test = '2021-08'

# %% [markdown]
# ## Carga de Datos

# %%
vectores = joblib.load(data_from + 'vectores.joblib')
nombres_targets = joblib.load(data_from + 'targets.joblib')
nombres_features = joblib.load(data_from + 'features.joblib')

nombres_targets['seccion'] = nombres_targets['seccion'].replace(
    {'sociedad': 0, 'economia': 1, 'el-mundo': 2}
)

# %% [markdown]
# ## Armado de Datasets

# %%
X_train_w1 = vectores[
    (nombres_targets['mes'] >= walk1_train[0])
    & (nombres_targets['mes'] <= walk1_train[1])
]
y_train_w1 = nombres_targets[
    (nombres_targets['mes'] >= walk1_train[0])
    & (nombres_targets['mes'] <= walk1_train[1])
]['seccion']

X_train_w2 = vectores[
    (nombres_targets['mes'] >= walk2_train[0])
    & (nombres_targets['mes'] <= walk2_train[1])
]
y_train_w2 = nombres_targets[
    (nombres_targets['mes'] >= walk2_train[0])
    & (nombres_targets['mes'] <= walk2_train[1])
]['seccion']

X_train_w3 = vectores[
    (nombres_targets['mes'] >= walk3_train[0])
    & (nombres_targets['mes'] <= walk3_train[1])
]
y_train_w3 = nombres_targets[
    (nombres_targets['mes'] >= walk3_train[0])
    & (nombres_targets['mes'] <= walk3_train[1])
]['seccion']

X_test_w1 = vectores[nombres_targets['mes'] == walk1_test]
y_test_w1 = nombres_targets[nombres_targets['mes'] == walk1_test]['seccion']

X_test_w2 = vectores[nombres_targets['mes'] == walk2_test]
y_test_w2 = nombres_targets[nombres_targets['mes'] == walk2_test]['seccion']

X_test_w3 = vectores[nombres_targets['mes'] == walk3_test]
y_test_w3 = nombres_targets[nombres_targets['mes'] == walk3_test]['seccion']

# %% [markdown]
# ## Modelo Baseline
#
# Este modelo consiste en un clasificador bayesiano con la totalidad del dataset

# %%
# Paso 1
nb1 = MultinomialNB()
nb1.fit(X_train_w1, y_train_w1)

nb1_score_train = nb1.score(X_train_w1, y_train_w1)
nb1_score_test = nb1.score(X_test_w1, y_test_w1)

# %%
# Paso 2
nb2 = MultinomialNB()
nb2.fit(X_train_w2, y_train_w2)

nb2_score_train = nb2.score(X_train_w2, y_train_w2)
nb2_score_test = nb2.score(X_test_w2, y_test_w2)

# %%
# Paso 3
nb3 = MultinomialNB()
nb3.fit(X_train_w3, y_train_w3)

nb3_score_train = nb3.score(X_train_w3, y_train_w3)
nb3_score_test = nb3.score(X_test_w3, y_test_w3)

# %% [markdown]
# ## Metricas

# %%
nb_metrics = {
    'paso_1': {
        'train_size': X_train_w1.shape[0],
        'test_size': X_test_w1.shape[0],
        'train_from': walk1_train[0],
        'train_to': walk1_train[1],
        'test_month': walk1_test,
        'train_accuracy': nb1_score_train,
        'test_accuracy': nb1_score_test,
    },
    'paso_2': {
        'train_size': X_train_w2.shape[0],
        'test_size': X_test_w2.shape[0],
        'train_from': walk2_train[0],
        'train_to': walk2_train[1],
        'test_month': walk2_test,
        'train_accuracy': nb2_score_train,
        'test_accuracy': nb2_score_test,
    },
    'paso_3': {
        'train_size': X_train_w3.shape[0],
        'test_size': X_test_w3.shape[0],
        'train_from': walk3_train[0],
        'train_to': walk3_train[1],
        'test_month': walk3_test,
        'train_accuracy': nb3_score_train,
        'test_accuracy': nb3_score_test,
    },
    'average_accuracy': {
        'train_accuracy': np.mean([nb1_score_train, nb2_score_train, nb3_score_train]),
        'test_accuracy': np.mean([nb1_score_test, nb2_score_test, nb3_score_test]),
    },
}

print(json.dumps(nb_metrics, indent=4))

# %%
pd.DataFrame(nb_metrics).to_csv(data_to + 'nb_results.csv')

# %%
# Paso 1
y_predicted1 = nb1.predict(X_test_w1)
cm1 = confusion_matrix(y_test_w1, y_predicted1, labels=nb1.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=nb1.classes_)
disp.plot()

print("Métricas paso_1\n\n" + classification_report(y_test_w1, y_predicted1))

# %%
# Paso 2
y_predicted2 = nb2.predict(X_test_w2)
cm2 = confusion_matrix(y_test_w2, y_predicted2, labels=nb2.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=nb2.classes_)
disp.plot()

print(f"Métricas paso_2\n\n" + classification_report(y_test_w2, y_predicted2))

# %%
# Paso 3
y_predicted3 = nb3.predict(X_test_w3)
cm3 = confusion_matrix(y_test_w3, y_predicted3, labels=nb3.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=nb3.classes_)
disp.plot()
print(f"Métricas paso_3\n\n" + classification_report(y_test_w3, y_predicted3))
