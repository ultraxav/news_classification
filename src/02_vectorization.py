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
# # Tratado de páginas web a datasets para entrenamiento
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
import pandas as pd

from utils.tokenizers import tokenizador
from sklearn.feature_extraction.text import CountVectorizer
from typing import List

# %% [markdown]
# ## Carga de datasets

# %%
df_el_mundo = pd.read_csv('../data/01_raw/df_noticias_el-mundo.csv')
df_economia = pd.read_csv('../data/01_raw/df_noticias_economia.csv')
df_sociedad = pd.read_csv('../data/01_raw/df_noticias_sociedad.csv')

df_el_mundo['fecha'] = pd.to_datetime(df_el_mundo['fecha'], format='%d/%m/%Y')
df_economia['fecha'] = pd.to_datetime(df_economia['fecha'], format='%d/%m/%Y')
df_sociedad['fecha'] = pd.to_datetime(df_sociedad['fecha'], format='%d/%m/%Y')

df_noticias = pd.concat([df_el_mundo, df_economia, df_sociedad])
print(f'{df_noticias.shape[0]} noticias totales')
df_noticias = df_noticias[pd.isna(df_noticias['nota']) == False]
print(f'{df_noticias.shape[0]} noticias sin notas sin texto')
df_noticias

# %%
df_el_mundo.iloc[34]

# %%
df_economia.iloc[34]

# %%
df_sociedad.iloc[34]

# %%
print('Fechas Mínimas')
print(df_noticias[df_noticias['seccion'] == 'el-mundo']['fecha'].min())
print(df_noticias[df_noticias['seccion'] == 'economia']['fecha'].min())
print(df_noticias[df_noticias['seccion'] == 'sociedad']['fecha'].min())

print('\nFechas Máximas')
print(df_noticias[df_noticias['seccion'] == 'el-mundo']['fecha'].max())
print(df_noticias[df_noticias['seccion'] == 'economia']['fecha'].max())
print(df_noticias[df_noticias['seccion'] == 'sociedad']['fecha'].max())

# %%
df_noticias = df_noticias[
    df_noticias['fecha']
    >= df_noticias[df_noticias['seccion'] == 'el-mundo']['fecha'].min()
]

# %%
print('Fechas Mínimas')
print(df_noticias[df_noticias['seccion'] == 'el-mundo']['fecha'].min())
print(df_noticias[df_noticias['seccion'] == 'economia']['fecha'].min())
print(df_noticias[df_noticias['seccion'] == 'sociedad']['fecha'].min())

print('\nFechas Máximas')
print(df_noticias[df_noticias['seccion'] == 'el-mundo']['fecha'].max())
print(df_noticias[df_noticias['seccion'] == 'economia']['fecha'].max())
print(df_noticias[df_noticias['seccion'] == 'sociedad']['fecha'].max())

# %% [markdown]
# ## Definiciones

# %%
# Storpwords en español
STOPWORDS_FILE = 'utils/stopwords_es.txt'
STOPWORDS_FILE_SIN_ACENTOS = 'utils/stopwords_es_sin_acentos.txt'

# Cantidad minima y maxima de docs que tienen que tener a un token para conservarlo.
MIN_DF = 3
MAX_DF = 0.8

# Numero minimo y maximo de tokens consecutivos que se consideran
MIN_NGRAMS = 1
MAX_NGRAMS = 2

# Nombre de datasets tratados
VECTORS_FILE = '../data/02_processed/vectores.joblib'
TARGETS_FILE = '../data/02_processed/targets.joblib'
FEATURE_NAMES_FILE = '../data/02_processed/features.joblib'


# %% [markdown]
# ## Funciones de apoyo

# %%
def leer_stopwords(path: str) -> List[str]:
    with open(path, 'rt') as stopwords_file:
        return [
            stopword
            for stopword in [stopword.strip().lower() for stopword in stopwords_file]
            if len(stopword) > 0
        ]


# %% [markdown]
# ## Generación de Datasets

# %%
mi_lista_stopwords = leer_stopwords(STOPWORDS_FILE_SIN_ACENTOS)

mi_tokenizer = tokenizador()

vectorizer = CountVectorizer(
    stop_words=mi_lista_stopwords,
    tokenizer=mi_tokenizer,
    lowercase=True,
    strip_accents='unicode',
    decode_error='ignore',
    ngram_range=(MIN_NGRAMS, MAX_NGRAMS),
    min_df=MIN_DF,
    max_df=MAX_DF,
)

# fit = tokenizar y codificar documentos como filas
todos_los_vectores = vectorizer.fit_transform(df_noticias['nota'])

# guardar vectores de docs y la correspondiente categoria asignada a cada doc.
joblib.dump(todos_los_vectores, VECTORS_FILE)

joblib.dump(df_noticias['seccion'], TARGETS_FILE)

print('Finalizado, el dataset está en {} y {}.'.format(VECTORS_FILE, TARGETS_FILE))

nombres_features = vectorizer.get_feature_names()

joblib.dump(nombres_features, FEATURE_NAMES_FILE)

print('El nombre de cada columna de features esta en {}.'.format(FEATURE_NAMES_FILE))
