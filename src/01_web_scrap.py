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
# # Scrapping de Páginas Web
#
# ## Integrantes:
#
# * Del Villar, Javier
# * Pistoya, Haydeé Soledad
# * Sorza, Andrés
#
# ## Cargamos Librerías

# %%
import json
import pandas as pd
import requests
import time
from tqdm import tqdm

from bs4 import BeautifulSoup

# %% [markdown]
# ## Difinir que sección del diario a scrappear

# %%
# seccion = 'el-mundo' # 350
# seccion = 'economia' # 400
seccion = 'sociedad'  # 800
pags = 800

data_path = '../data/01_raw/'

# %% [markdown]
# ## Estructura de la página
#
# La página contiene ld+json que contienen los links a las noticias, se procede a extraerlos

# %%
response = requests.get('https://www.pagina12.com.ar/secciones/' + seccion)
soup = BeautifulSoup(response.content, 'html.parser')

# %%
ldjson = soup.find_all("script", type="application/ld+json")
parsedjson = json.loads(ldjson[2].contents[0])
parsedjson

# %%
lista_url = []
for item_url in parsedjson['itemListElement']:
    lista_url.append(item_url['url'])

lista_url

# %% [markdown]
# ## Scrap de links de noticias

# %%
df_links = pd.DataFrame(columns=['pag', 'link'])

df_links = pd.DataFrame(
    {
        'pag': [0] * len(lista_url),
        'link': lista_url,
    }
)

df_links

# %%
df_links.to_csv(data_path + 'df_links_' + seccion + '.csv', index=False)

# %%
for i in tqdm(range(1, pags, 1)):
    df_links = None
    lista_url = []

    response = requests.get(
        'https://www.pagina12.com.ar/secciones/' + seccion + '?page=' + str(i)
    )
    soup = BeautifulSoup(response.content, 'html.parser')

    ldjson = soup.find_all("script", type="application/ld+json")
    parsedjson = json.loads(ldjson[2].contents[0])

    for item_url in parsedjson['itemListElement']:
        lista_url.append(item_url['url'])

    df_links = pd.DataFrame(
        {
            'pag': [i] * len(lista_url),
            'link': lista_url,
        }
    )

    df_links.to_csv(
        '../data/01_raw/df_links_' + seccion + '.csv',
        mode='a',
        header=False,
        index=False,
    )

    time.sleep(3)

# %% [markdown]
# ## Scrap de noticias

# %%
response = requests.get(
    'https://www.pagina12.com.ar/364811-internet-de-alta-velocidad-al-alcance-de-los-ciudadanos'
)
soup = BeautifulSoup(response.content, 'html.parser')

# %%
soup.title.string

# %%
df_noticia = {
    'titulo': soup.title.string.split(' |')[0],
    'fecha': soup.find_all('div', class_='date')[1].string,
    'nota': soup.find('div', class_='article-main-content article-text').get_text(
        separator=' ', strip=True
    ),
    'link': 'https://www.pagina12.com.ar/364811-internet-de-alta-velocidad-al-alcance-de-los-ciudadanos',
    'seccion': seccion,
}

df_noticia

# %%
df_noticias = pd.DataFrame(columns=['titulo', 'fecha', 'nota', 'link', 'seccion'])
df_noticias.to_csv(data_path + 'df_noticias_' + seccion + '.csv', index=False)

df_error = pd.DataFrame(columns=['link', 'seccion'])
df_error.to_csv(data_path + 'df_error_' + seccion + '.csv', index=False)

noticias = pd.read_csv(data_path + 'df_links_' + seccion + '.csv')

noticias = noticias['link']

# %%
for noticia in tqdm(noticias):
    try:
        response = requests.get(noticia)
        soup = BeautifulSoup(response.content, 'html.parser')

        df_noticia = (
            pd.Series(
                {
                    'titulo': soup.title.string.split(' |')[0],
                    'fecha': soup.find_all('div', class_='date')[1].string,
                    'nota': soup.find(
                        'div', class_='article-main-content article-text'
                    ).get_text(separator=' ', strip=True),
                    'link': noticia,
                    'seccion': seccion,
                }
            )
            .to_frame()
            .transpose()
        )

        df_noticia.to_csv(
            data_path + 'df_noticias_' + seccion + '.csv',
            mode='a',
            header=False,
            index=False,
        )

    except:
        df_error = (
            pd.Series(
                {
                    'link': noticia,
                    'seccion': seccion,
                }
            )
            .to_frame()
            .transpose()
        )

        df_error.to_csv(
            data_path + 'df_error_' + seccion + '.csv',
            mode='a',
            header=False,
            index=False,
        )

    time.sleep(3)
