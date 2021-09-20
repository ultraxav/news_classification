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
# El objetivo de este notebook es entrenar un modelo SVC para clasificación de noticias, se experimentarán con distintas herramientas para conocer el performance y las potenciabilidades de dicho modelo, tal cuales son:
#
# * Selección de variables
# * Entrenamiento
# * Optimización Bayesiana
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
import optuna
import pandas as pd

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm

data_from = '../data/02_processed/'
data_to = '../data/03_models/'

seed = 1234

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

# %%
len(nombres_features)

# %% [markdown]
# ## Armado de Datasets y entrenamiento de Modelo Baseline
#
# Entrenamiento de un SVC con todas las variables como punto de referencia.
#
# Datos de entrenamiento desde 2020-07 hasta 2020-10

# %%
# %%time
svc_baseline_metrics = {}

i = 4

X_train = vectores[nombres_targets['mes'].isin(meses[:i])]
y_train = nombres_targets[nombres_targets['mes'].isin(meses[:i])]['seccion']

svc = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=seed)
svc.fit(X_train, y_train)

svc_baseline_metrics['train_base'] = {}

svc_baseline_metrics['train_base']['train_with'] = str(i) + ' months'
svc_baseline_metrics['train_base']['train_from'] = str(meses[0])
svc_baseline_metrics['train_base']['train_to'] = str(meses[i])

svc_baseline_metrics['train_base']['train_size'] = X_train.shape[0]
svc_baseline_metrics['train_base']['train_score'] = svc.score(X_train, y_train)

test_scores = []
for j in range(len(meses[:i]), len(meses), 1):
    X_test = vectores[nombres_targets['mes'] == meses[j]]
    y_test = nombres_targets[nombres_targets['mes'] == meses[j]]['seccion']

    test_scores.append(svc.score(X_test, y_test))

svc_baseline_metrics['train_base']['test_scores'] = test_scores

svc_baseline_metrics['train_base']['test_score_mean'] = np.mean(test_scores)

svc_baseline_metrics['train_base']['C'] = svc.get_params()['C']
svc_baseline_metrics['train_base']['k_features'] = X_train.shape[1]

# %%
print(json.dumps(svc_baseline_metrics, indent=4))

# %%
# train
y_predicted_train = svc.predict(X_train)
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
y_predicted_test = svc.predict(X_test)
cm2 = confusion_matrix(y_test, y_predicted_test, normalize='true')
disp2 = ConfusionMatrixDisplay(
    confusion_matrix=cm2, display_labels=['sociedad', 'economia', 'el-mundo']
)
disp2.plot()

print(
    f'Métricas en mes más lejano ({meses[-1]}):\n\n'
    + classification_report(y_test, y_predicted_test)
)

# %% [markdown]
# ## Importancia de Variables
#
# ### 1ra vuelta
#
# De 1% a 100% de las variables

# %%
df_results_1 = pd.DataFrame(columns=['percent_selected', 'accuracy'])

# %%
for i in tqdm(range(1, 101, 1)):
    svc = SVC(
        kernel='linear', probability=True, class_weight='balanced', random_state=seed
    )

    percent = i / 100

    selector_features = SelectKBest(
        score_func=chi2, k=int(len(nombres_features) * percent)
    )
    selector_features.fit(X_train, y_train)

    train_selected = selector_features.transform(X_train)
    test_selected = selector_features.transform(X_test)

    selector_features.get_support()

    svc.fit(train_selected, y_train)

    acc_score = svc.score(test_selected, y_test)

    result = {
        'percent_selected': i,
        'accuracy': acc_score,
    }

    df_results_1 = df_results_1.append(result, ignore_index=True)


# %%
plt.figure()
plt.xlabel('Percent of features selected')
plt.ylabel('Accuracy')
plt.plot('percent_selected', 'accuracy', data=df_results_1)
plt.show()

# %% [markdown]
# ### 2da vuelta
#
# De 0.01% a 1% de las variables

# %%
df_results_2 = pd.DataFrame(columns=['percent_selected', 'accuracy'])

# %%
for i in tqdm(range(1, 100, 1)):
    svc = SVC(
        kernel='linear', probability=True, class_weight='balanced', random_state=seed
    )

    percent = i / 10000

    selector_features = SelectKBest(
        score_func=chi2, k=int(len(nombres_features) * percent)
    )
    selector_features.fit(X_train, y_train)

    train_selected = selector_features.transform(X_train)
    test_selected = selector_features.transform(X_test)

    selector_features.get_support()

    svc.fit(train_selected, y_train)

    acc_score = svc.score(test_selected, y_test)

    result = {
        'percent_selected': percent,
        'accuracy': acc_score,
    }

    df_results_2 = df_results_2.append(result, ignore_index=True)

# %%
plt.figure()
plt.xlabel('Percent of features selected')
plt.ylabel('Accuracy')
plt.plot('percent_selected', 'accuracy', data=df_results_2)
plt.show()

# %%
df_results = pd.concat([df_results_1, df_results_2])
df_results = df_results.sort_values(by='percent_selected').reset_index(drop=True)
df_results

# %%
plt.figure()
plt.xlabel('Percent of features selected')
plt.ylabel('Accuracy')
plt.plot('percent_selected', 'accuracy', data=df_results)
plt.show()

# %%
df_results.to_csv(data_to + 'df_feature_results.csv', index=False)

# %% [markdown]
# # Optimización Bayesiana

# %%
learner = SVC(
    kernel='linear', probability=True, class_weight='balanced', random_state=seed
)

# Hiperparámetro a optimizar
param = {'C': optuna.distributions.LogUniformDistribution(1e-10, 1e10)}

# Inicialización del experimento
svc_search = optuna.integration.OptunaSearchCV(
    learner,
    param,
    cv=StratifiedKFold(),
    n_jobs=-1,
    n_trials=100,
    random_state=seed,
    study=optuna.create_study(
        study_name='news_svc',
        direction='maximize',
        storage='sqlite:///' + data_to + 'news_svc.db',
    ),
)

# %%
selector_features = SelectKBest(score_func=chi2, k=880)
selector_features.fit(X_train, y_train)

train_selected_w1 = selector_features.transform(X_train)
test_selected_w1 = selector_features.transform(X_test)

# %%
# %%time
svc_search.fit(train_selected_w1, y_train)

# %%
study = svc_search.study
train_params = svc_search.best_params_
train_params

# %% [markdown]
# # Entrenamiento de de modelos final

# %%
# Datasets
X_train = vectores[nombres_targets['mes'].isin(meses[:4])]
y_train = nombres_targets[nombres_targets['mes'].isin(meses[:4])]['seccion']

selector_features = SelectKBest(score_func=chi2, k=880)
selector_features.fit(X_train, y_train)

vectores = selector_features.transform(vectores)

# %%
vectores.shape

# %%
train_selected = vectores[nombres_targets['mes'].isin(meses[:4])]
train_selected

# %%
# %%time
svc_metrics = {}

i = 4

svc = SVC(
    C=train_params['C'],
    kernel='linear',
    probability=True,
    class_weight='balanced',
    random_state=seed,
)

svc.fit(train_selected, y_train)

svc_metrics['train4'] = {}

svc_metrics['train4']['train_with'] = str(i) + ' months'
svc_metrics['train4']['train_from'] = str(meses[0])
svc_metrics['train4']['train_to'] = str(meses[i])

svc_metrics['train4']['train_size'] = train_selected.shape[0]
svc_metrics['train4']['train_score'] = svc.score(train_selected, y_train)

test_scores = []
for j in range(len(meses[:i]), len(meses), 1):
    X_test = vectores[nombres_targets['mes'] == meses[j]]
    y_test = nombres_targets[nombres_targets['mes'] == meses[j]]['seccion']

    test_scores.append(svc.score(X_test, y_test))

svc_metrics['train4']['test_scores'] = test_scores

svc_metrics['train4']['test_score_mean'] = np.mean(test_scores)

svc_metrics['train4']['C'] = train_params['C']
svc_metrics['train4']['k_features'] = train_selected.shape[1]

# %%
print(json.dumps(svc_metrics, indent=4))

# %%
with open(data_to + 'svc_metrics.json', 'w') as fp:
    json.dump({**svc_baseline_metrics, **svc_metrics}, fp, indent=4)

# %% [markdown]
# ## Degradación del Accuracy

# %%
with open(data_to + 'nb_metrics.json', 'r') as fp:
    nb_metrics = json.load(fp)

# %%
# data
plt.plot(nb_metrics['train4']['test_scores'], label='nb_baseline')
plt.plot(svc_baseline_metrics['train_base']['test_scores'], label='svc_baseline')
plt.plot(svc_metrics['train4']['test_scores'], label='svc_train_final')

# format
plt.title('Degradación del Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1)
plt.xlabel('Cantidad de meses después del entrenamiento')

# plot
plt.legend()
plt.show()

# %%
# train
y_predicted_train = svc.predict(train_selected)
cm1 = confusion_matrix(y_train, y_predicted_train, normalize='true')
disp1 = ConfusionMatrixDisplay(
    confusion_matrix=cm1, display_labels=['sociedad', 'economia', 'el-mundo']
)
disp1.plot()

print(
    f'Métricas en Entrenamiento\n\n' + classification_report(y_train, y_predicted_train)
)

# %%
X_test = vectores[nombres_targets['mes'] == meses[-1]]
y_test = nombres_targets[nombres_targets['mes'] == meses[-1]]['seccion']

# Test on farthest away month
y_predicted_test = svc.predict(X_test)
cm2 = confusion_matrix(y_test, y_predicted_test, normalize='true')
disp2 = ConfusionMatrixDisplay(
    confusion_matrix=cm2, display_labels=['sociedad', 'economia', 'el-mundo']
)
disp2.plot()

print(
    f'Métricas en mes más lejano ({meses[-1]}):\n\n'
    + classification_report(y_test, y_predicted_test)
)

# %%
