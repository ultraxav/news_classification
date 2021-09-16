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
# Entrenamiento de un SVC con todas las variables, y en distintas ventanas de tiempo

# %%
# Paso 1
svc1 = SVC(
    kernel='linear', probability=True, class_weight='balanced', random_state=seed
)
svc1.fit(X_train_w1, y_train_w1)

svc1_score_train = svc1.score(X_train_w1, y_train_w1)
svc1_score_test = svc1.score(X_test_w1, y_test_w1)

# %%
# Paso 2
svc2 = SVC(
    kernel='linear', probability=True, class_weight='balanced', random_state=seed
)
svc2.fit(X_train_w2, y_train_w2)

svc2_score_train = svc2.score(X_train_w2, y_train_w2)
svc2_score_test = svc2.score(X_test_w2, y_test_w2)

# %%
# Paso 3
svc3 = SVC(
    kernel='linear', probability=True, class_weight='balanced', random_state=seed
)
svc3.fit(X_train_w3, y_train_w3)

svc3_score_train = svc3.score(X_train_w3, y_train_w3)
svc3_score_test = svc3.score(X_test_w3, y_test_w3)

# %%
svc_baseline_metrics = {
    'paso_1': {
        'train_size': X_train_w1.shape[0],
        'test_size': X_test_w1.shape[0],
        'train_from': walk1_train[0],
        'train_to': walk1_train[1],
        'test_month': walk1_test,
        'train_accuracy': svc1_score_train,
        'test_accuracy': svc1_score_test,
    },
    'paso_2': {
        'train_size': X_train_w2.shape[0],
        'test_size': X_test_w2.shape[0],
        'train_from': walk2_train[0],
        'train_to': walk2_train[1],
        'test_month': walk2_test,
        'train_accuracy': svc2_score_train,
        'test_accuracy': svc2_score_test,
    },
    'paso_3': {
        'train_size': X_train_w3.shape[0],
        'test_size': X_test_w3.shape[0],
        'train_from': walk3_train[0],
        'train_to': walk3_train[1],
        'test_month': walk3_test,
        'train_accuracy': svc3_score_train,
        'test_accuracy': svc3_score_test,
    },
    'average_accuracy': {
        'train_accuracy': np.mean(
            [svc1_score_train, svc2_score_train, svc3_score_train]
        ),
        'test_accuracy': np.mean([svc1_score_test, svc2_score_test, svc3_score_test]),
    },
}

print(json.dumps(svc_baseline_metrics, indent=4))

# %%
pd.DataFrame(svc_baseline_metrics).to_csv(data_to + 'svc_baseline_metrics.csv')

# %%
# Paso 1
y_predicted1 = svc1.predict(X_test_w1)
cm1 = confusion_matrix(y_test_w1, y_predicted1, labels=svc1.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm1, display_labels=['sociedad', 'economia', 'el-mundo']
)
disp.plot()

print("Métricas paso_1\n\n" + classification_report(y_test_w1, y_predicted1))

# %%
# Paso 2
y_predicted2 = svc1.predict(X_test_w2)
cm2 = confusion_matrix(y_test_w2, y_predicted2, labels=svc2.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm2, display_labels=['sociedad', 'economia', 'el-mundo']
)
disp.plot()

print("Métricas paso_2\n\n" + classification_report(y_test_w2, y_predicted2))

# %%
# Paso 3
y_predicted3 = svc1.predict(X_test_w3)
cm3 = confusion_matrix(y_test_w3, y_predicted3, labels=svc3.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm3, display_labels=['sociedad', 'economia', 'el-mundo']
)
disp.plot()

print("Métricas paso_3\n\n" + classification_report(y_test_w3, y_predicted3))

# %% [markdown]
# ## Importancia de Variables
#
# ### 1ra vuelta

# %%
df_results_1 = pd.DataFrame(columns=['percent_selected', 'accuracy'])

# %%
for i in tqdm(range(1, 101, 1)):
    svc1 = SVC(
        kernel='linear', probability=True, class_weight='balanced', random_state=seed
    )
    percent = i / 100
    selector_features = SelectKBest(
        score_func=chi2, k=int(len(nombres_features) * percent)
    )
    selector_features.fit(X_train_w1, y_train_w1)

    train_selected = selector_features.transform(X_train_w1)
    test_selected = selector_features.transform(X_test_w1)

    selector_features.get_support()

    svc1.fit(train_selected, y_train_w1)

    acc_score = svc1.score(test_selected, y_test_w1)

    result = {
        'percent_selected': i,
        'accuracy': acc_score,
    }

    df_results_1 = df_results_1.append(result, ignore_index=True)


# %%
# df_results_1.to_csv(data_to + 'df_feature_results_1.csv', index=False)

# %%
plt.figure()
plt.xlabel('Percent of features selected')
plt.ylabel('Accuracy')
plt.plot('percent_selected', 'accuracy', data=df_results_1)
plt.show()

# %% [markdown]
# ### 2da vuelta

# %%
df_results_2 = pd.DataFrame(columns=['percent_selected', 'accuracy'])

# %%
for i in tqdm(range(1, 100, 1)):
    svc1 = SVC(
        kernel='linear', probability=True, class_weight='balanced', random_state=seed
    )
    percent = i / 1000
    selector_features = SelectKBest(
        score_func=chi2, k=int(len(nombres_features) * percent)
    )
    selector_features.fit(X_train_w1, y_train_w1)

    train_selected = selector_features.transform(X_train_w1)
    test_selected = selector_features.transform(X_test_w1)

    selector_features.get_support()

    svc1.fit(train_selected, y_train_w1)

    acc_score = svc1.score(test_selected, y_test_w1)

    result = {
        'percent_selected': percent,
        'accuracy': acc_score,
    }

    df_results_2 = df_results_2.append(result, ignore_index=True)

# %%
# df_results_2.to_csv(data_to + 'df_feature_results_2.csv', index=False)

# %%
plt.figure()
plt.xlabel('Percent of features selected')
plt.ylabel('Accuracy')
plt.plot('percent_selected', 'accuracy', data=df_results_2)
plt.show()

# %% [markdown]
# ### 3ra vuelta

# %%
df_results_3 = pd.DataFrame(columns=['percent_selected', 'accuracy'])

# %%
for i in tqdm(range(1, 100, 1)):
    svc1 = SVC(
        kernel='linear', probability=True, class_weight='balanced', random_state=seed
    )
    percent = i / 10000
    selector_features = SelectKBest(
        score_func=chi2, k=int(len(nombres_features) * percent)
    )
    selector_features.fit(X_train_w1, y_train_w1)

    train_selected = selector_features.transform(X_train_w1)
    test_selected = selector_features.transform(X_test_w1)

    selector_features.get_support()

    svc1.fit(train_selected, y_train_w1)

    acc_score = svc1.score(test_selected, y_test_w1)

    result = {
        'percent_selected': percent,
        'accuracy': acc_score,
    }

    df_results_3 = df_results_3.append(result, ignore_index=True)

# %%
# df_results_3.to_csv(data_to + 'df_feature_results_3.csv', index=False)

# %%
plt.figure()
plt.xlabel('Percent of features selected')
plt.ylabel('Accuracy')
plt.plot('percent_selected', 'accuracy', data=df_results_3)
plt.show()

# %%
df_results = pd.concat([df_results_1, df_results_2, df_results_3])
df_results = df_results.sort_values(by='percent_selected').reset_index(drop=True)
df_results

# %%
plt.figure(figsize=(15, 10))
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
        study_name='news_svc', direction='maximize', storage='sqlite:///news_svc.db'
    ),
)

# %%
selector_features = SelectKBest(score_func=chi2, k=500)
selector_features.fit(X_train_w1, y_train_w1)

train_selected_w1 = selector_features.transform(X_train_w1)
test_selected_w1 = selector_features.transform(X_test_w1)

# %%
# %%time
svc_search.fit(train_selected_w1, y_train_w1)

# %%
study = svc_search.study
train_params = svc_search.best_params_
train_params

# %% [markdown]
# # Entrenamiento de de modelos finales

# %%
# Paso 1
selector_features = SelectKBest(score_func=chi2, k=500)
selector_features.fit(X_train_w1, y_train_w1)

train_selected_w1 = selector_features.transform(X_train_w1)
test_selected_w1 = selector_features.transform(X_test_w1)

svc1 = SVC(
    C=train_params['C'],
    kernel='linear',
    probability=True,
    class_weight='balanced',
    random_state=seed,
)
svc1.fit(train_selected_w1, y_train_w1)

svc1_score_train = svc1.score(train_selected_w1, y_train_w1)
svc1_score_test = svc1.score(test_selected_w1, y_test_w1)

# %%
# Paso 2
selector_features = SelectKBest(score_func=chi2, k=500)
selector_features.fit(X_train_w2, y_train_w2)

train_selected_w2 = selector_features.transform(X_train_w2)
test_selected_w2 = selector_features.transform(X_test_w2)

svc2 = SVC(
    C=train_params['C'],
    kernel='linear',
    probability=True,
    class_weight='balanced',
    random_state=seed,
)
svc2.fit(train_selected_w2, y_train_w2)

svc2_score_train = svc2.score(train_selected_w2, y_train_w2)
svc2_score_test = svc2.score(test_selected_w2, y_test_w2)

# %%
# Paso 3
selector_features = SelectKBest(score_func=chi2, k=500)
selector_features.fit(X_train_w3, y_train_w3)

train_selected_w3 = selector_features.transform(X_train_w3)
test_selected_w3 = selector_features.transform(X_test_w3)

svc3 = SVC(
    C=train_params['C'],
    kernel='linear',
    probability=True,
    class_weight='balanced',
    random_state=seed,
)
svc3.fit(train_selected_w3, y_train_w3)

svc3_score_train = svc3.score(train_selected_w3, y_train_w3)
svc3_score_test = svc3.score(test_selected_w3, y_test_w3)

# %%
svc_final_metrics = {
    'paso_1': {
        'train_size': X_train_w1.shape[0],
        'test_size': X_test_w1.shape[0],
        'train_from': walk1_train[0],
        'train_to': walk1_train[1],
        'test_month': walk1_test,
        'train_accuracy': svc1_score_train,
        'test_accuracy': svc1_score_test,
    },
    'paso_2': {
        'train_size': X_train_w2.shape[0],
        'test_size': X_test_w2.shape[0],
        'train_from': walk2_train[0],
        'train_to': walk2_train[1],
        'test_month': walk2_test,
        'train_accuracy': svc2_score_train,
        'test_accuracy': svc2_score_test,
    },
    'paso_3': {
        'train_size': X_train_w3.shape[0],
        'test_size': X_test_w3.shape[0],
        'train_from': walk3_train[0],
        'train_to': walk3_train[1],
        'test_month': walk3_test,
        'train_accuracy': svc3_score_train,
        'test_accuracy': svc3_score_test,
    },
    'average_accuracy': {
        'train_accuracy': np.mean(
            [svc1_score_train, svc2_score_train, svc3_score_train]
        ),
        'test_accuracy': np.mean([svc1_score_test, svc2_score_test, svc3_score_test]),
    },
}

print(json.dumps(svc_final_metrics, indent=4))

# %%
pd.DataFrame(svc_final_metrics).to_csv(data_to + 'svc_final_metrics.csv')

# %%
# Paso 1
y_predicted1 = svc1.predict(test_selected_w1)
cm1 = confusion_matrix(y_test_w1, y_predicted1, labels=svc1.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm1, display_labels=['sociedad', 'economia', 'el-mundo']
)
disp.plot()

print("Métricas paso_1\n\n" + classification_report(y_test_w1, y_predicted1))

# %%
# Paso 2
y_predicted2 = svc2.predict(test_selected_w2)
cm2 = confusion_matrix(y_test_w2, y_predicted2, labels=svc2.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm2, display_labels=['sociedad', 'economia', 'el-mundo']
)
disp.plot()

print("Métricas paso_2\n\n" + classification_report(y_test_w2, y_predicted2))

# %%
# Paso 3
y_predicted3 = svc3.predict(test_selected_w3)
cm3 = confusion_matrix(y_test_w3, y_predicted3, labels=svc3.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm3, display_labels=['sociedad', 'economia', 'el-mundo']
)
disp.plot()

print("Métricas paso_3\n\n" + classification_report(y_test_w3, y_predicted3))

# %%
