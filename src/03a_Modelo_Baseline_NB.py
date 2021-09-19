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
import matplotlib.pyplot as plt
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

# %% [markdown]
# ## Carga de Datos

# %%
vectores = joblib.load(data_from + 'vectores.joblib')
nombres_targets = joblib.load(data_from + 'targets.joblib')
nombres_features = joblib.load(data_from + 'features.joblib')

nombres_targets['seccion'] = nombres_targets['seccion'].replace(
    {'sociedad': 0, 'economia': 1, 'el-mundo': 2}
)

meses = nombres_targets['mes'].unique()
meses

# %% [markdown]
# ## Armado de Datasets y entrenamiento de Modelo Baseline
#
# Este modelo consiste en un clasificador bayesiano con la totalidad del dataset

# %%
nb_metrics = {}

for i in range(3, 7, 1):
    X_train = vectores[nombres_targets['mes'].isin(meses[:i])]
    y_train = nombres_targets[nombres_targets['mes'].isin(meses[:i])]['seccion']

    nbc = MultinomialNB()
    nbc.fit(X_train, y_train)

    nb_metrics['train' + str(i)] = {}

    nb_metrics['train' + str(i)]['train_with'] = str(i) + ' months'
    nb_metrics['train' + str(i)]['train_from'] = str(meses[0])
    nb_metrics['train' + str(i)]['train_to'] = str(meses[i])

    nb_metrics['train' + str(i)]['train_size'] = X_train.shape[0]
    nb_metrics['train' + str(i)]['train_score'] = nbc.score(X_train, y_train)

    test_scores = []
    for j in range(len(meses[:i]), len(meses), 1):
        X_test = vectores[nombres_targets['mes'] == meses[j]]
        y_test = nombres_targets[nombres_targets['mes'] == meses[j]]['seccion']

        test_scores.append(nbc.score(X_test, y_test))

    nb_metrics['train' + str(i)]['test_scores'] = test_scores

    nb_metrics['train' + str(i)]['test_score_mean'] = np.mean(test_scores)

# %% [markdown]
# ## Metricas

# %%
with open(data_to + 'nb_metrics.json', 'w') as fp:
    json.dump(nb_metrics, fp, indent=4)

print(json.dumps(nb_metrics, indent=4))

# %% [markdown]
# ## Degradación del Accuracy

# %%
# data
plt.plot(nb_metrics['train3']['test_scores'], label='train3')
plt.plot(nb_metrics['train4']['test_scores'], label='train4')
plt.plot(nb_metrics['train5']['test_scores'], label='train5')
plt.plot(nb_metrics['train6']['test_scores'], label='train6')

# format
plt.title('Degradación del Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1)
plt.xlabel('Cantidad de meses después del entrenamiento')

# plot
plt.legend()
plt.show()

# %% [markdown]
# ## Entrenamiento de modelo final

# %%
X_train = vectores[nombres_targets['mes'].isin(meses[:4])]
y_train = nombres_targets[nombres_targets['mes'].isin(meses[:4])]['seccion']

X_test = vectores[nombres_targets['mes'] == meses[-1]]
y_test = nombres_targets[nombres_targets['mes'] == meses[-1]]['seccion']

nbc = MultinomialNB()
nbc.fit(X_train, y_train)

# %%
# train
y_predicted_train = nbc.predict(X_train)
cm1 = confusion_matrix(y_train, y_predicted_train, normalize='true')
disp1 = ConfusionMatrixDisplay(
    confusion_matrix=cm1, display_labels=['sociedad', 'economia', 'el-mundo']
)
disp1.plot()

print(
    f'Métricas en Entrenamiento\n\n' + classification_report(y_train, y_predicted_train)
)

# %%
# Test on farthest away month
y_predicted_test = nbc.predict(X_test)
cm2 = confusion_matrix(y_test, y_predicted_test, normalize='true')
disp2 = ConfusionMatrixDisplay(
    confusion_matrix=cm2, display_labels=['sociedad', 'economia', 'el-mundo']
)
disp2.plot()

print(
    f'Métricas en mes más lejano ({meses[-1]}):\n\n'
    + classification_report(y_test, y_predicted_test)
)
