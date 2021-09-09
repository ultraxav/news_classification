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
# # SVC Model - Feature importance and selection, Optimization and Training
#
# Fechas de venatana temporal 1:
# * Desde: 2020-07
# * Hasta: 2021-01
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

# import matplotlib.pyplot as plt
# import pandas as pd
# import sklearn
# import numpy as np

# from sklearn.metrics import log_loss
# from sklearn.svm import SVC
# from sklearn.feature_selection import SequentialFeatureSelector

data_from = '../data/02_processed/'
data_to = '../data/03_models/'
ventana = '1'

train_date = '2020-07'
test_date = '2021-01'

seed = 1234

# %% [markdown]
# ## Carga de Datos

# %%
vectores = joblib.load(data_from + 'vectores_' + ventana + '.joblib')
nombres_targets = joblib.load(data_from + 'targets_' + ventana + '.joblib')
nombres_features = joblib.load(data_from + 'features_' + ventana + '.joblib')

nombres_targets['seccion'] = nombres_targets['seccion'].replace(
    {'sociedad': 0, 'economia': 1, 'el-mundo': 2}
)

Xtrain = vectores[nombres_targets['mes'] <= train_date]
ytrain = nombres_targets[nombres_targets['mes'] <= train_date]['seccion']

# %%
len(nombres_features) * 0.10

# %% [markdown]
# # Recursive Feature Selection
#
# El objetivo de esta sección es explorar la eliminación recursiva de features que no aportan a la ganancia final del modelo.
# Se utlilizarán únicamente los datos de train.
#
# > Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through any specific attribute or callable. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached. **Scikit-Learn**
#
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

# %%
# Se crea un modelo sin entonar, el razonamiento es el siguiente, si una variable es altamente predictora, figurará como predictora aún cuando el modelo es sencillo
learner = SVC(
    kernel='linear', probability=True, class_weight='balanced', random_state=seed
)

# Continuar la búsuqeda hasta quedarse con una sola variable
min_features_to_select = 1

# Catidad de variables a eliminar en cada ronda (Se seleccionó aprox. el 10%)
step = 20000

rfecv_svc = RFECV(
    estimator=learner,
    step=step,
    cv=StratifiedKFold(3, random_state=seed),
    scoring='neg_log_loss',
    min_features_to_select=min_features_to_select,
)

rfecv_svc.fit(Xtrain, ytrain)

print(f'Optimal number of features: {rfecv_svc.n_features_}')

# %%
# %%time
rfecv_svc.fit(X, y)

# %% [markdown]
# inspección visual y seleccion de cantidad de variables importantes

# %%
plt.figure(figsize=(10, 8))
plt.xlabel('Number of features selected')
plt.ylabel('Cross validation score (Negative Log Loss)')
plt.plot(
    range(min_features_to_select, len(rfecv_svc.grid_scores_) + min_features_to_select),
    rfecv_svc.grid_scores_,
)
plt.show()
