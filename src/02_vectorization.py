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

data_from = '../data/01_raw/'
data_to = '../data/02_processed/'
# ventana, date_from, date_to = ['1', '2020-07', '2021-01']
ventana, date_from, date_to = ['2', '2021-02', '2021-08']

# %% [markdown]
# ## Carga de datasets

# %%
df_el_mundo = pd.read_csv(data_from + 'df_noticias_el-mundo.csv')
df_economia = pd.read_csv(data_from + 'df_noticias_economia.csv')
df_sociedad = pd.read_csv(data_from + 'df_noticias_sociedad.csv')

df_el_mundo['fecha'] = pd.to_datetime(df_el_mundo['fecha'], format='%d/%m/%Y')
df_economia['fecha'] = pd.to_datetime(df_economia['fecha'], format='%d/%m/%Y')
df_sociedad['fecha'] = pd.to_datetime(df_sociedad['fecha'], format='%d/%m/%Y')

df_noticias = pd.concat([df_el_mundo, df_economia, df_sociedad])
print(f'{df_noticias.shape[0]} noticias totales')
df_noticias = df_noticias[pd.isna(df_noticias['nota']) == False]
print(f'{df_noticias.shape[0]} noticias sin notas sin texto')
df_noticias

# %%
df_noticias = df_noticias.sort_values(by='fecha').reset_index(drop=True)
print('Fechas Mínimas')
print(df_noticias[df_noticias['seccion'] == 'el-mundo']['fecha'].min())
print(df_noticias[df_noticias['seccion'] == 'economia']['fecha'].min())
print(df_noticias[df_noticias['seccion'] == 'sociedad']['fecha'].min())

# %%
df_noticias['mes'] = df_noticias['fecha'].dt.to_period('M')

# %%
df_noticias.pivot_table(
    index='mes', columns='seccion', values='nota', aggfunc='count'
).sort_index()

# %%
df_noticias = df_noticias[df_noticias['mes'] >= date_from]
df_noticias = df_noticias[df_noticias['mes'] <= date_to]

# %%
df_noticias.pivot_table(
    index='mes', columns='seccion', values='nota', aggfunc='count'
).sort_index()

# %%
df_noticias['seccion'].value_counts()

# %%
df_noticias = df_noticias.reset_index(drop=True)

# %%
print('Fechas Mínimas')
print(df_noticias['fecha'].min())

print('\nFechas Máximas')
print(df_noticias['fecha'].max())

print(f'\n{df_noticias.shape[0]} noticias filtradas por fecha')

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
VECTORS_FILE = data_to + 'vectores_' + ventana + '.joblib'
TARGETS_FILE = data_to + 'targets_' + ventana + '.joblib'
FEATURE_NAMES_FILE = data_to + 'features_' + ventana + '.joblib'


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

joblib.dump(df_noticias[['mes', 'seccion']], TARGETS_FILE)

nombres_features = vectorizer.get_feature_names()

joblib.dump(nombres_features, FEATURE_NAMES_FILE)

print(f'Finalizado! los datasets se encuentran en la carpeta {data_to}')
