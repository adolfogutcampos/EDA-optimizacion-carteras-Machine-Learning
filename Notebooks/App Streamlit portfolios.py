#!/usr/bin/env python
# coding: utf-8

# # Indice:
# 
# [0.- Introduccion](#0--introduccion)
# 
# [1.- Importar librerias necesarias](#1--importar-librerias-necesarias)
# 
#    - [1.1.- Importar datasets de archivos del repositorio](#11--importar-datasets-de-archivos-del-repositorio)
# 
# [2.- Principios del EDA (Exploratory Data Analysis)](#2--principios-del-eda-exploratory-data-analysis)
# 
#    - [2.1.- Contexto](#21--contexto)
# 
#    - [2.2.- Hipotesis](#22--hipotesis)
# 
# [3.- Cotizacion y rentabilidad de conjunto de 35 empresas y 10 indices](#3--cotizacion-y-rentabilidad-de-conjunto-de-35-empresas-y-10-indices)
# 
#    - [3.1.- Cotizaciones](#31--cotizaciones)
# 
#      - [3.1.1.- Cotizaciones de las 35 empresas](#311--cotizaciones-de-las-35-empresas)
# 
#      - [3.1.2.- Cotizaciones de los 10 indices bursatiles](#312--cotizaciones-de-los-10-indices-bursatiles)
# 
#    - [3.2.- Rentabilidades](#32--rentabilidades)
# 
#      - [3.2.1.- Rentabilidad de 35 empresas](#321--rentabilidad-de-35-empresas)
# 
#      - [3.2.2.- Rentabilidad de 10 indices](#322--rentabilidad-de-10-indices)
# 
# [4.- Seleccion de muestra de empresas e indices para una cartera de 10 valores](#4--seleccion-de-muestra-de-empresas-e-indices-para-una-cartera-de-10-valores)
# 
# [5.- Cartera optimizada con Sharpe](#5--cartera-optimizada-con-sharpe)
# 
#    - [5.1.- Teoria](#51--teoria)
# 
#    - [5.2.- Portfolio con cartera optimizada sin modelos de machine learning (Modelo 1)](#52--portfolio-con-cartera-optimizada-sin-modelos-de-machine-learning-modelo-1)
# 
#    - [5.3.- Portfolio con cartera optimizada con simulacion de Monte Carlo (Modelo 2)](#53--portfolio-con-cartera-optimizada-con-simulacion-de-monte-carlo-modelo-2)
# 
# [6.- Carteras optimizadas con Modelos de Machine Learning](#6--carteras-optimizadas-con-modelos-de-machine-learning)
# 
#    - [6.1.- Modelo Supervisado ML con Gradient Boosting (Modelo 3)](#61--modelo-supervisado-ml-con-gradient-boosting-modelo-3)
# 
#    - [6.2.- Modelo Supervisado ML con XGBoost (Modelo 4)](#62--modelo-supervisado-ml-con-xgboost-modelo-4)
# 
#    - [6.3.- Modelo No Supervisado PCA (Principal Component Analysis) (Modelo 5)](#63--modelo-no-supervisado-pca-principal-component-analysis-modelo-5)
# 
# [7.- Comparacion de resultados finales de los 5 modelos](#7--comparacion-de-resultados-finales-de-los-5-modelos)
# 
#    - [7.1.- Comparacion de retorno, volatilidad y sharpe de las 5 carteras](#71--comparacion-de-retorno-volatilidad-y-sharpe-de-las-5-carteras)
#    
#    - [7.2.- Comparacion de peso de carteras](#72--comparacion-de-peso-de-carteras)
# 
# 

# # 0.- Introduccion
# 
# El análisis de datos financieros es una herramienta crucial en el mundo de las inversiones. Proporciona información valiosa sobre el comportamiento del mercado y ayuda a los inversores a tomar decisiones informadas. 
# 
# Este Análisis Exploratorio de Datos (EDA) se centra en las 35 empresas que conforman el IBEX 35 y 10 índices bursátiles globales. El objetivo principal es optimizar carteras de inversión mediante diversas técnicas y modelos, evaluando su rendimiento en términos de retorno, volatilidad y ratio Sharpe.
# 
# De entre las 35 empresas y 10 indices, se tomará 5 empresas y 5 indices para desarrollar carteras optimizadas.

# # 1.- Importar librerias necesarias

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf

import plotly as pl
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px

from scipy import stats
from scipy.stats import skew, kurtosis

import warnings
warnings.filterwarnings('ignore')


# ## 1.1.- Importar datasets de archivos del repositorio

# Aquí vamos a realizar la lectura (pd.read_csv) de los datasets de las 35 empresas.

# In[2]:


#1 Acciona
ANA_dataset = pd.read_csv('../data/ANA_dataset.csv')
#2 Acciona Energías
ANE_dataset = pd.read_csv('../data/ANE_dataset.csv')
#3 ACS
ACS_dataset = pd.read_csv('../data/ACS_dataset.csv')
#4 Acerinox
ACX_dataset = pd.read_csv('../data/ACX_dataset.csv')
#5 Aena
AENA_dataset = pd.read_csv('../data/AENA_dataset.csv')
#6 Amadeus
AMS_dataset = pd.read_csv('../data/AMS_dataset.csv')
#7 ArcelorMittal
MTS_dataset = pd.read_csv('../data/MTS_dataset.csv')
#8 Banco Sabadell
SAB_dataset = pd.read_csv('../data/SAB_dataset.csv')
#9 Banco Santander
SAN_dataset = pd.read_csv('../data/SAN_dataset.csv')
#10 Bankinter
BKT_dataset = pd.read_csv('../data/BKT_dataset.csv')
#11 BBVA
BBVA_dataset = pd.read_csv('../data/BBVA_dataset.csv')
#12 CaixaBank
CABK_dataset = pd.read_csv('../data/CABK_dataset.csv')
#13 Cellnex
CLNX_dataset = pd.read_csv('../data/CLNX_dataset.csv')
#14 Colonial
COL_dataset = pd.read_csv('../data/COL_dataset.csv')
#15 Enagás
ENG_dataset = pd.read_csv('../data/ENG_dataset.csv')
#16 Endesa	
ELE_dataset = pd.read_csv('../data/ELE_dataset.csv')
#17 Ferrovial
FER_dataset = pd.read_csv('../data/FER_dataset.csv')
#18 Fluidra
FDR_dataset = pd.read_csv('../data/FDR_dataset.csv')
#19 Grifols
GRF_dataset = pd.read_csv('../data/GRF_dataset.csv')
#20 IAG	
IAG_dataset = pd.read_csv('../data/IAG_dataset.csv')
#21 Iberdrola
IBE_dataset = pd.read_csv('../data/IBE_dataset.csv')
#22 Indra
IDR_dataset = pd.read_csv('../data/IDR_dataset.csv')
#23 Inditex
ITX_dataset = pd.read_csv('../data/ITX_dataset.csv')
#24 Logista
LOG_dataset = pd.read_csv('../data/LOG_dataset.csv')
#25 Mapfre
MAP_dataset = pd.read_csv('../data/MAP_dataset.csv')
#26 Meliá
MEL_dataset = pd.read_csv('../data/MEL_dataset.csv')
#27 Merlin Properties
MRL_dataset = pd.read_csv('../data/MRL_dataset.csv')
#28 Naturgy
NTGY_dataset = pd.read_csv('../data/NTGY_dataset.csv')
#29 Redeia
RED_dataset = pd.read_csv('../data/RED_dataset.csv')
#30 Repsol
REP_dataset = pd.read_csv('../data/REP_dataset.csv')
#31 Rovi
ROVI_dataset = pd.read_csv('../data/ROVI_dataset.csv')
#32 Sacyr
SCYR_dataset = pd.read_csv('../data/SCYR_dataset.csv')
#33 Solaria
SLR_dataset = pd.read_csv('../data/SLR_dataset.csv')
#34 Telefónica
TEF_dataset = pd.read_csv('../data/TEF_dataset.csv')
#35 Unicaja
UNI_dataset = pd.read_csv('../data/UNI_dataset.csv')


# In[3]:


#1 NIFTY50
NIFTY50_dataset = pd.read_csv('../data/NIFTY50_dataset.csv')
#2 ASX200
ASX200_dataset = pd.read_csv('../data/ASX200_dataset.csv')
#3 S&P 500
SP500_dataset = pd.read_csv('../data/SP500_dataset.csv')
#4 Dow Jones
DowJones_dataset = pd.read_csv('../data/DowJones_dataset.csv')
#5 Nasdaq
Nasdaq_dataset = pd.read_csv('../data/Nasdaq_dataset.csv')
#6 S&P/TSX Composite
SPCanada_dataset = pd.read_csv('../data/SPCanada_dataset.csv')
#7 DAX 40
DAX40_dataset = pd.read_csv('../data/DAX40_dataset.csv')
#8 FTSE 100
FTSE100_dataset = pd.read_csv('../data/FTSE100_dataset.csv')
#9 Eurostoxx 50
Euro50_dataset = pd.read_csv('../data/Euro50_dataset.csv')
#10 IBEX 35
IBEX35_dataset = pd.read_csv('../data/IBEX35_dataset.csv')


# De entre el listado de las 35 empresas, vamos a ver las columnas de ACS como ejemplo:

# In[4]:


ACS_dataset = pd.read_csv('../data/ACS_dataset.csv')
ACS_dataset


# De los 10 indices bursátiles, vamos a ver las columnas del indice del NIFTY50 como ejemplo:

# In[5]:


NIFTY50_dataset = pd.read_csv('../data/NIFTY50_dataset.csv')
NIFTY50_dataset


# # 2.- Principios del EDA (Exploratory Data Analysis)

# ## 2.1.- Contexto

# El IBEX 35 es el principal índice bursátil de la Bolsa de Madrid, que agrupa a las 35 empresas más líquidas y de mayor capitalización de mercado de España. Además, los índices bursátiles seleccionados para este análisis, como el S&P 500, NASDAQ, y NIFTY 50, representan los principales mercados financieros a nivel mundial. 
# 
# El análisis de estos datos permite una visión amplia del comportamiento del mercado tanto a nivel local como global.

# ## 2.2.- Hipotesis
# 
# 1- **Diversificación**: ¿Puede una cartera diversificada reducir la volatilidad general y mejorar el ratio Sharpe?
# 
# 2- **Eficiencia de Mercado**: ¿Cómo impacta la inclusión de valores de distintos mercados globales en la estabilidad y el rendimiento de la cartera?
# 
# 3- **Modelos Supervisados**: ¿Pueden los modelos de machine learning supervisados (Gradient Boosting y XGBoost) identificar patrones complejos en los datos históricos de precios y mejorar la optimización de la cartera?
# 
# 4- **Simulación de Monte Carlo**: ¿Generar múltiples combinaciones de pesos aleatorios puede proporcionar configuraciones óptimas sin necesidad de un modelo supervisado?
# 
# 5- **Reducción de Dimensionalidad con PCA**: ¿La utilización de PCA para reducir la dimensionalidad del problema puede simplificar la optimización de la cartera sin perder información crucial?

# # 3.- Cotizacion y rentabilidad de conjunto de 35 empresas y 10 indices

# ## 3.1.- Cotizaciones

# En el apartado de cotizaciones, se va a proceder a separar los gráficos de cotizaciones de las 35 empresas y el de los 10 índices bursátiles. Esto es debido a que las empresas del IBEX 35 cotizan en euros y tienen a tener una valoración alrededor de máximo 100 euros. Mientras que la cotización de los índices bursátiles se muestra en puntos, y su cotización puede ser de miles de puntos.
# 
# En otras palabras, se realiza una separación de gráficos para diferenciar un gráfico por divisa en EUR, y diferenciar otro gráfico basado en puntos.

# ### 3.1.1.- Cotizaciones de las 35 empresas

# In[6]:


# Lista de archivos CSV y nombres de columnas correspondientes
files = [
    'ANA_dataset.csv', 'ANE_dataset.csv', 'ACS_dataset.csv', 'ACX_dataset.csv', 'AENA_dataset.csv',
    'AMS_dataset.csv', 'MTS_dataset.csv', 'SAB_dataset.csv', 'SAN_dataset.csv', 'BKT_dataset.csv',
    'BBVA_dataset.csv', 'CABK_dataset.csv', 'CLNX_dataset.csv', 'COL_dataset.csv', 'ENG_dataset.csv',
    'ELE_dataset.csv', 'FER_dataset.csv', 'FDR_dataset.csv', 'GRF_dataset.csv', 'IAG_dataset.csv',
    'IBE_dataset.csv', 'IDR_dataset.csv', 'ITX_dataset.csv', 'LOG_dataset.csv', 'MAP_dataset.csv',
    'MEL_dataset.csv', 'MRL_dataset.csv', 'NTGY_dataset.csv', 'RED_dataset.csv', 'REP_dataset.csv',
    'ROVI_dataset.csv', 'SCYR_dataset.csv', 'SLR_dataset.csv', 'TEF_dataset.csv', 'UNI_dataset.csv'
]


# In[7]:


company_names = [
    'ANA', 'ANE', 'ACS', 'ACX', 'AENA', 'AMS', 'MTS', 'SAB', 'SAN', 'BKT', 
    'BBVA', 'CABK', 'CLNX', 'COL', 'ENG', 'ELE', 'FER', 'FDR', 'GRF', 'IAG', 
    'IBE', 'IDR', 'ITX', 'LOG', 'MAP', 'MEL', 'MRL', 'NTGY', 'RED', 'REP', 
    'ROVI', 'SCYR', 'SLR', 'TEF', 'UNI'
]


# In[8]:


dataframes = []
for file, name in zip(files, company_names):
    df = pd.read_csv(f'../data/{file}')
    df['Company'] = name
    dataframes.append(df)


# In[9]:


all_data = pd.concat(dataframes, ignore_index=True)


# In[10]:


all_data['Date'] = pd.to_datetime(all_data['Date'])


# In[11]:


fig = px.line(
    all_data,
    x='Date',
    y='Close',
    color='Company',
    title='Precio de cierre (Close) en el período 01/01/2018 - 20/05/2024',
    labels={'Close': 'Precio de cierre (EUR)', 'Date': 'Fecha'}
)

fig.show()


# ### 3.1.2.- Cotizaciones de los 10 indices bursatiles

# In[12]:


index_files = [
    'NIFTY50_dataset.csv', 'ASX200_dataset.csv', 'SP500_dataset.csv', 'DowJones_dataset.csv',
    'Nasdaq_dataset.csv', 'SPCanada_dataset.csv', 'DAX40_dataset.csv', 'FTSE100_dataset.csv',
    'Euro50_dataset.csv', 'IBEX35_dataset.csv'
]


# In[13]:


index_names = [
    'NIFTY50', 'ASX200', 'SP500', 'DowJones', 'Nasdaq',
    'SPCanada', 'DAX40', 'FTSE100', 'Euro50', 'IBEX35'
]


# In[14]:


index_dataframes = []
for file, name in zip(index_files, index_names):
    df = pd.read_csv(f'../data/{file}')
    df['Index'] = name
    index_dataframes.append(df)


# In[15]:


all_index_data = pd.concat(index_dataframes, ignore_index=True)


# In[16]:


all_index_data['Date'] = pd.to_datetime(all_index_data['Date'])


# In[17]:


fig = px.line(
    all_index_data,
    x='Date',
    y='Close',
    color='Index',
    title='Valor de cierre de los 10 índices entre 01/01/2018 y 20/05/2024',
    labels={'Close': 'Valor de cierre (puntos)', 'Date': 'Fechas'}
)

fig.show()


# ## 3.2.- Rentabilidades

# ### 3.2.1.- Rentabilidad de 35 empresas

# In[18]:


# Lista para almacenar las rentabilidades
returns = []


# In[19]:


# Calcular la rentabilidad para cada empresa
for file, name in zip(files, company_names):
    df = pd.read_csv(f'../data/{file}')
    initial_close = df['Close'].iloc[0]
    final_close = df['Close'].iloc[-1]
    return_percentage = ((final_close - initial_close) / initial_close) * 100
    returns.append({'Company': name, 'Return': return_percentage})


# In[20]:


returns_df = pd.DataFrame(returns)


# In[21]:


# Ordenar el DataFrame por la columna 'Return' en orden descendente
returns_df = returns_df.sort_values(by='Return', ascending=False)


# In[22]:


fig = px.bar(
    returns_df,
    x='Company',
    y='Return',
    color='Return',
    title='Rentabilidad de 35 empresas del IBEX 35 entre 01/01/2018 y 20/05/2024',
    labels={'Return': 'Rentabilidad (%)', 'Company': 'Empresas'},
    hover_data=['Return'],
    color_continuous_scale='Viridis'  
)

fig.update_layout(xaxis_tickangle=-45)

fig.show()


# ### 3.2.2.- Rentabilidad de 10 indices

# In[23]:


returns = []


# In[24]:


for file, name in zip(index_files, index_names):
    df = pd.read_csv(f'../data/{file}')
    initial_close = df['Close'].iloc[0]
    final_close = df['Close'].iloc[-1]
    return_percentage = ((final_close - initial_close) / initial_close) * 100
    returns.append({'Index': name, 'Return': return_percentage})


# In[25]:


returns_df = pd.DataFrame(returns)


# In[26]:


returns_df = returns_df.sort_values(by='Return', ascending=False)


# In[27]:


fig = px.bar(
    returns_df,
    x='Index',
    y='Return',
    color='Return',
    title='Rentabilidad de 10 índices bursátiles entre 01/01/2018 y 20/05/2024',
    labels={'Return': 'Rentabilidad (%)', 'Index': 'Indices Bursátiles'},
    hover_data=['Return'],
    color_continuous_scale='Viridis'  
)

fig.update_layout(xaxis_tickangle=-45)

fig.show()


# # 4.- Seleccion de muestra de empresas e indices para una cartera de 10 valores

# Es aquí cuando vamos a escoger una empresa con el fin de analizarla de forma individual.
# 
# Para ello, vamos a crear una variable llamada dataset_empresa, la cual va a contener el dataset de la empresa que elijamos de entre las 35.
# 
# Como ejemplo, vamos a escoger el dataset de ACS. Generando una variable paralela mediante:
# 
# dataset_empresa = ACS_dataset
# 
# No obstante, podríamos cambiar el dataset al de otra empresa para realizar el analisis, cambiando en la siguiente celda las siglas de ACS por las de otra empresa. Como por ejemplo: ITX, quedando así como resultado: 
# 
# dataset_empresa = ITX_dataset

# En definitiva, la cartera que se va a generar está compuesta por los siguientes valores:

# In[28]:


#Empresas dentro de la cartera

#1 ACS
ACS_dataset
#2 Banco Santander
SAN_dataset
#3 BBVA
BBVA_dataset
#4 Repsol
REP_dataset
#5 Iberdrola
IBE_dataset

#Indices dentro de la cartera

#6 NIFTY50 (India)
NIFTY50_dataset
#7 ASX200 (Australia)
ASX200_dataset
#8 S&P 500 (EEUU)
SP500_dataset
#9 Nasdaq (EEUU)
Nasdaq_dataset
#10 S&P/TSX Composite (Canada)
SPCanada_dataset


# # 5.- Cartera optimizada con Sharpe

# ## 5.1.- Teoria

# ## 5.2.- Portfolio con cartera optimizada sin modelos de machine learning (Modelo 1)

# In[29]:


from scipy.optimize import minimize


# Paso 1: Calcular retorno y volatilidad para cada valor
# 
# Calculamos el retorno anualizado y la volatilidad anualizada para cada valor. Utilizamos la media y la desviación estándar de los cambios porcentuales diarios (retornos diarios) multiplicados por 252 (el número aproximado de días de trading en un año).

# In[30]:


def calcular_retorno_volatilidad(datos):
    retorno = datos['Close'].pct_change().mean() * 252
    volatilidad = datos['Close'].pct_change().std() * np.sqrt(252)
    return retorno, volatilidad


# In[31]:


valores = {
    'ACS': ACS_dataset,
    'SAN': SAN_dataset,
    'BBVA': BBVA_dataset,
    'REP': REP_dataset,
    'IBE': IBE_dataset,
    'NIFTY50': NIFTY50_dataset,
    'ASX200': ASX200_dataset,
    'SP500': SP500_dataset,
    'Nasdaq': Nasdaq_dataset,
    'SPCanada': SPCanada_dataset
}


# In[32]:


retorno_volatilidad = {}
for nombre, datos in valores.items():
    retorno_volatilidad[nombre] = calcular_retorno_volatilidad(datos)


# Paso 2: Mostrar retorno y volatilidad de cada valor
# 
# Mostramos los retornos y las volatilidades calculadas para cada valor.

# In[33]:


for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    print(f'{nombre}: Retorno={retorno}, Volatilidad={volatilidad}')


# Paso 3: Crear matriz de covarianza
# 
# La matriz de covarianza mide cómo los retornos de los diferentes valores se mueven juntos. Esta matriz es crucial para calcular la volatilidad de la cartera.

# In[34]:


matriz_covarianza = np.zeros((len(valores), len(valores)))
for i, (_, datos_i) in enumerate(valores.items()):
    for j, (_, datos_j) in enumerate(valores.items()):
        matriz_covarianza[i][j] = datos_i['Close'].pct_change().cov(datos_j['Close'].pct_change()) * 252


# Paso 4: Función objetivo para minimizar la volatilidad de la cartera
# 
# Definimos una función objetivo que calcula la volatilidad de la cartera en función de los pesos asignados a cada valor. Esta función se utilizará en la optimización.

# In[35]:


def objetivo(pesos):
    return np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))


# Paso 5: Restricciones para los pesos de la cartera
# 
# Definimos una restricción que asegura que la suma de los pesos sea igual a 1, es decir, que se invierta el 100% del capital.

# In[36]:


restricciones = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1})


# Paso 6: Optimizar la cartera para maximizar el ratio Sharpe
# 
# Utilizamos el método de optimización SLSQP para encontrar los pesos que minimizan la volatilidad, respetando la restricción definida.

# In[37]:


pesos_iniciales = np.ones(len(valores)) / len(valores)
resultados = minimize(objetivo, pesos_iniciales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones)


# Paso 7: Calcular retorno, volatilidad y ratio Sharpe de la cartera óptima
# 
# Calculamos el retorno esperado, la volatilidad y el ratio Sharpe de la cartera utilizando los pesos óptimos encontrados.

# In[38]:


retorno_cartera = np.dot([retorno for _, retorno in retorno_volatilidad.values()], resultados.x)
volatilidad_cartera = np.sqrt(np.dot(resultados.x.T, np.dot(matriz_covarianza, resultados.x)))
sharpe_cartera = retorno_cartera / volatilidad_cartera


# Paso 8: Mostrar resultado
# 
# Mostramos el retorno, la volatilidad y el ratio Sharpe de la cartera óptima.

# In[39]:


print(f'Retorno de la cartera: {retorno_cartera}')
print(f'Volatilidad de la cartera: {volatilidad_cartera}')
print(f'Ratio Sharpe de la cartera: {sharpe_cartera}')


# Paso 9: Mostrar pesos de la cartera óptima
# 
# Mostramos los pesos asignados a cada valor en la cartera óptima.

# In[40]:


pesos_cartera_optima = resultados.x * 100
print('Pesos de la cartera óptima:')
for nombre, peso in zip(valores.keys(), pesos_cartera_optima):
    print(f'{nombre}: {peso:.2f}%')


# Paso 10: Calcular y mostrar la frontera eficiente de Markowitz
# 
# Calculamos la frontera eficiente, que representa las carteras que ofrecen el máximo retorno para un nivel dado de riesgo.

# In[41]:


frontera_volatilidad = np.linspace(0, 0.3, 100)
frontera_retorno = []
for volatilidad in frontera_volatilidad:
    restricciones_frontera = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1},
                              {'type': 'eq', 'fun': lambda pesos: np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos))) - volatilidad})
    resultados_frontera = minimize(lambda pesos: -np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos), pesos_iniciales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones_frontera)
    retorno = np.dot([retorno for _, retorno in retorno_volatilidad.values()], resultados_frontera.x)
    frontera_retorno.append(retorno)


# Paso 11: Graficar la frontera eficiente de Markowitz
# 
# Creamos un gráfico para mostrar la frontera eficiente y los puntos de los valores individuales

# In[42]:


fig = go.Figure()

# Añadir puntos de valores individuales
for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    fig.add_trace(go.Scatter(x=[volatilidad], y=[retorno], mode='markers', name=nombre))

# Añadir la frontera eficiente de Markowitz
fig.add_trace(go.Scatter(x=frontera_volatilidad, y=frontera_retorno, mode='lines', name='Frontera eficiente de Markowitz', line=dict(color='red', dash='dash')))

fig.update_layout(title='Frontera eficiente de Markowitz',
                  xaxis_title='Volatilidad',
                  yaxis_title='Retorno esperado')

fig.show()


# Paso 12: Crear gráfico interactivo circular con los pesos de la cartera óptima
# 
# Creamos un gráfico circular para visualizar los pesos de la cartera óptima.

# In[43]:


pesos_cartera_optima_pct = resultados.x * 100
pesos_labels = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_cartera_optima_pct)]
fig_pie = go.Figure(data=[go.Pie(labels=pesos_labels, values=pesos_cartera_optima_pct, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
fig_pie.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
fig_pie.update_layout(title='Pesos de la cartera óptima', showlegend=False)
fig_pie.show()


# ### Resultados finales modelo 1:

# In[44]:


print(f'Retorno de la cartera: {retorno_cartera}')
print(f'Volatilidad de la cartera: {volatilidad_cartera}')
print(f'Ratio Sharpe de la cartera: {sharpe_cartera}')


# In[45]:


retorno_cartera_modelo_1 = retorno_cartera
volatilidad_cartera_modelo_1 = volatilidad_cartera
sharpe_cartera_modelo_1 = sharpe_cartera


# In[49]:


retorno_cartera_modelo_1


# In[47]:


volatilidad_cartera_modelo_1


# In[48]:


sharpe_cartera_modelo_1


# In[51]:


pesos_cartera_optima_modelo_1 = {nombre: peso for nombre, peso in zip(valores.keys(), pesos_cartera_optima)}
pesos_cartera_optima_modelo_1


# ## 5.3.- Portfolio con cartera optimizada con simulacion de Monte Carlo (Modelo 2)

# In[44]:


from scipy.optimize import minimize


# In[45]:


ACS_dataset = pd.read_csv('../data/ACS_dataset.csv')
SAN_dataset = pd.read_csv('../data/SAN_dataset.csv')
BBVA_dataset = pd.read_csv('../data/BBVA_dataset.csv')
REP_dataset = pd.read_csv('../data/REP_dataset.csv')
IBE_dataset = pd.read_csv('../data/IBE_dataset.csv')
NIFTY50_dataset = pd.read_csv('../data/NIFTY50_dataset.csv')
ASX200_dataset = pd.read_csv('../data/ASX200_dataset.csv')
SP500_dataset = pd.read_csv('../data/SP500_dataset.csv')
Nasdaq_dataset = pd.read_csv('../data/Nasdaq_dataset.csv')
SPCanada_dataset = pd.read_csv('../data/SPCanada_dataset.csv')


# Paso 1: Calcular Retorno y Volatilidad para Cada Valor
# 
# En este paso, calculamos el retorno anualizado y la volatilidad anualizada para cada uno de los valores. Utilizamos la media y la desviación estándar de los cambios porcentuales diarios (retornos diarios) multiplicados por 252 (el número aproximado de días de trading en un año).

# In[52]:


# Función para calcular el retorno y la volatilidad
def calcular_retorno_volatilidad(datos):
    retorno = datos['Close'].pct_change().mean() * 252
    volatilidad = datos['Close'].pct_change().std() * np.sqrt(252)
    return retorno, volatilidad


# In[53]:


# Datasets de los valores seleccionados
valores = {
    'ACS': ACS_dataset,
    'SAN': SAN_dataset,
    'BBVA': BBVA_dataset,
    'REP': REP_dataset,
    'IBE': IBE_dataset,
    'NIFTY50': NIFTY50_dataset,
    'ASX200': ASX200_dataset,
    'SP500': SP500_dataset,
    'Nasdaq': Nasdaq_dataset,
    'SPCanada': SPCanada_dataset
}


# In[54]:


# Calcular retorno y volatilidad
retorno_volatilidad = {nombre: calcular_retorno_volatilidad(datos) for nombre, datos in valores.items()}


# In[55]:


for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    print(f'{nombre}: Retorno={retorno}, Volatilidad={volatilidad}')


# Paso 2: Crear la Matriz de Covarianza
# 
# La matriz de covarianza mide cómo se mueven juntos los retornos de los diferentes valores. Es crucial para calcular la volatilidad de la cartera.

# In[56]:


# Crear matriz de covarianza
matriz_covarianza = np.zeros((len(valores), len(valores)))
for i, (_, datos_i) in enumerate(valores.items()):
    for j, (_, datos_j) in enumerate(valores.items()):
        matriz_covarianza[i][j] = datos_i['Close'].pct_change().cov(datos_j['Close'].pct_change()) * 252


# Paso 3: Simulación de Montecarlo (Podemos cambiar el número de 10000 simulaciones)
# 
# En este paso, generamos 10,000 carteras aleatorias (podemos cambiar este número). Para cada cartera, calculamos el retorno esperado, la volatilidad esperada y el ratio Sharpe.

# In[57]:


num_simulaciones = 10000
resultados = np.zeros((num_simulaciones, len(valores) + 3))


# In[58]:


for i in range(num_simulaciones):
    pesos = np.random.random(len(valores))
    pesos /= np.sum(pesos)
    
    retorno_esperado = np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos)
    volatilidad_esperada = np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))
    ratio_sharpe = retorno_esperado / volatilidad_esperada
    
    resultados[i, :len(valores)] = pesos
    resultados[i, len(valores)] = retorno_esperado
    resultados[i, len(valores) + 1] = volatilidad_esperada
    resultados[i, len(valores) + 2] = ratio_sharpe


# Paso 4: Encontrar la Cartera con el Ratio Sharpe Máximo y Mínimo
# 
# Aquí identificamos las carteras con el ratio Sharpe máximo y mínimo entre las simuladas. El ratio Sharpe mide el rendimiento ajustado por el riesgo.

# In[59]:


# Encontrar la cartera con el ratio Sharpe máximo
indice_sharpe_max = np.argmax(resultados[:, len(valores) + 2])
pesos_sharpe_max = resultados[indice_sharpe_max, :len(valores)]
retorno_sharpe_max = resultados[indice_sharpe_max, len(valores)]
volatilidad_sharpe_max = resultados[indice_sharpe_max, len(valores) + 1]
sharpe_max = resultados[indice_sharpe_max, len(valores) + 2]


# In[60]:


# Encontrar la cartera con el ratio Sharpe mínimo
indice_sharpe_min = np.argmin(resultados[:, len(valores) + 2])
pesos_sharpe_min = resultados[indice_sharpe_min, :len(valores)]
retorno_sharpe_min = resultados[indice_sharpe_min, len(valores)]
volatilidad_sharpe_min = resultados[indice_sharpe_min, len(valores) + 1]
sharpe_min = resultados[indice_sharpe_min, len(valores) + 2]


# Paso 5: Mostrar Resultados
# 
# Se presentan los resultados, incluyendo los pesos de cada valor en las carteras óptimas (tanto la de Sharpe máximo como la de Sharpe mínimo), así como los respectivos retornos y volatilidades.

# In[61]:


print(f'Ratio Sharpe Máximo: {sharpe_max}')
print(f'Retorno de la Cartera (Sharpe Máximo): {retorno_sharpe_max}')
print(f'Volatilidad de la Cartera (Sharpe Máximo): {volatilidad_sharpe_max}')
print('Pesos de la Cartera (Sharpe Máximo):')
for nombre, peso in zip(valores.keys(), pesos_sharpe_max):
    print(f'{nombre}: {peso:.2f}%')


# In[62]:


print(f'Ratio Sharpe Mínimo: {sharpe_min}')
print(f'Retorno de la Cartera (Sharpe Mínimo): {retorno_sharpe_min}')
print(f'Volatilidad de la Cartera (Sharpe Mínimo): {volatilidad_sharpe_min}')
print('Pesos de la Cartera (Sharpe Mínimo):')
for nombre, peso in zip(valores.keys(), pesos_sharpe_min):
    print(f'{nombre}: {peso:.2f}%')


# Paso 6: Frontera Eficiente de Markowitz
# 
# Calculamos la frontera eficiente de Markowitz, que representa las carteras que ofrecen el máximo retorno para un nivel dado de riesgo.

# In[63]:


frontera_volatilidad = np.linspace(0, 0.3, 100)
frontera_retorno = []
for volatilidad in frontera_volatilidad:
    restricciones_frontera = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1},
                              {'type': 'eq', 'fun': lambda pesos: np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos))) - volatilidad})
    resultados_frontera = minimize(lambda pesos: -np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos), pesos_iniciales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones_frontera)
    retorno = np.dot([retorno for _, retorno in retorno_volatilidad.values()], resultados_frontera.x)
    frontera_retorno.append(retorno)


# Paso 7: Graficar la Frontera Eficiente y los Resultados
# 
# Graficamos los 10,000 portafolios generados, los puntos de los valores individuales y la frontera eficiente de Markowitz.

# In[64]:


fig = go.Figure()

# Añadir puntos de los portafolios de Montecarlo
fig.add_trace(go.Scatter(
    x=resultados[:, len(valores) + 1],
    y=resultados[:, len(valores)],
    mode='markers',
    marker=dict(color=resultados[:, len(valores) + 2], colorscale='Viridis', showscale=True, colorbar=dict(title='Ratio Sharpe')),
    name='Portafolios',
    legendgroup='Portafolios',
    showlegend=False  
))

# Añadir puntos de valores individuales
for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    fig.add_trace(go.Scatter(
        x=[volatilidad], y=[retorno], mode='markers', name=nombre,
        legendgroup='Valores Individuales', marker=dict(size=10)
    ))

# Añadir la frontera eficiente de Markowitz
fig.add_trace(go.Scatter(
    x=frontera_volatilidad, y=frontera_retorno, mode='lines', name='Frontera eficiente de Markowitz',
    line=dict(color='red', dash='dash'), legendgroup='Frontera'
))

fig.update_layout(
    title='10.000 Portafolios Simulados y Frontera Eficiente de Markowitz',
    xaxis_title='Volatilidad',
    yaxis_title='Retorno esperado',
    showlegend=True,
    width=1200,  
    height=800,  
    legend=dict(
        orientation="h",  
        yanchor="bottom", 
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.show()


# In[65]:


fig = go.Figure()

# Añadir puntos de valores individuales
for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    fig.add_trace(go.Scatter(x=[volatilidad], y=[retorno], mode='markers', name=nombre))

# Añadir la frontera eficiente de Markowitz
fig.add_trace(go.Scatter(x=frontera_volatilidad, y=frontera_retorno, mode='lines', name='Frontera eficiente de Markowitz', line=dict(color='red', dash='dash')))

fig.update_layout(title='Frontera eficiente de Markowitz',
                  xaxis_title='Volatilidad',
                  yaxis_title='Retorno esperado')

fig.show()


# Paso 8: Crear Gráfico Interactivo Circular con los Pesos de la Cartera Óptima
# 
# Finalmente, creamos un gráfico circular para visualizar los pesos de la cartera óptima (con el ratio Sharpe máximo).

# In[66]:


pesos_labels_max = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_sharpe_max * 100)]
fig_pie_max = go.Figure(data=[go.Pie(labels=pesos_labels_max, values=pesos_sharpe_max * 100, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
fig_pie_max.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
fig_pie_max.update_layout(title='Pesos de la Cartera Óptima (Sharpe Máximo)', showlegend=False)
fig_pie_max.show()


# ### Resultados finales modelo 2:

# In[67]:


retorno_cartera_modelo_2 = retorno_sharpe_max
volatilidad_cartera_modelo_2 = volatilidad_sharpe_max
sharpe_cartera_modelo_2 = sharpe_max


# In[68]:


retorno_cartera_modelo_2


# In[69]:


volatilidad_cartera_modelo_2


# In[70]:


sharpe_cartera_modelo_2


# In[71]:


pesos_cartera_optima_modelo_2 = {nombre: peso for nombre, peso in zip(valores.keys(), pesos_sharpe_max)}
pesos_cartera_optima_modelo_2


# # 6.- Carteras optimizadas con Modelos de Machine Learning

# ## 6.1.- Modelo Supervisado ML con Gradient Boosting (Modelo 3)

# In[72]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# In[73]:


ACS_dataset = pd.read_csv('../data/ACS_dataset.csv')
SAN_dataset = pd.read_csv('../data/SAN_dataset.csv')
BBVA_dataset = pd.read_csv('../data/BBVA_dataset.csv')
REP_dataset = pd.read_csv('../data/REP_dataset.csv')
IBE_dataset = pd.read_csv('../data/IBE_dataset.csv')
NIFTY50_dataset = pd.read_csv('../data/NIFTY50_dataset.csv')
ASX200_dataset = pd.read_csv('../data/ASX200_dataset.csv')
SP500_dataset = pd.read_csv('../data/SP500_dataset.csv')
Nasdaq_dataset = pd.read_csv('../data/Nasdaq_dataset.csv')
SPCanada_dataset = pd.read_csv('../data/SPCanada_dataset.csv')


# Paso 1: Calcular Retorno y Volatilidad para Cada Valor
# 
# Calculamos el retorno anualizado y la volatilidad anualizada para cada uno de los 10 valores. Utilizamos la media y la desviación estándar de los cambios porcentuales diarios (retornos diarios) multiplicados por 252 (el número aproximado de días de trading en un año).

# In[74]:


# funcion calcular el retorno y la volatilidad
def calcular_retorno_volatilidad(datos):
    retorno = datos['Close'].pct_change().mean() * 252
    volatilidad = datos['Close'].pct_change().std() * np.sqrt(252)
    return retorno, volatilidad


# In[75]:


valores = {
    'ACS': ACS_dataset,
    'SAN': SAN_dataset,
    'BBVA': BBVA_dataset,
    'REP': REP_dataset,
    'IBE': IBE_dataset,
    'NIFTY50': NIFTY50_dataset,
    'ASX200': ASX200_dataset,
    'SP500': SP500_dataset,
    'Nasdaq': Nasdaq_dataset,
    'SPCanada': SPCanada_dataset
}


# In[76]:


# Calcular retorno y volatilidad
retorno_volatilidad = {nombre: calcular_retorno_volatilidad(datos) for nombre, datos in valores.items()}


# In[77]:


for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    print(f'{nombre}: Retorno={retorno}, Volatilidad={volatilidad}')


# Paso 2: Crear la Matriz de Covarianza
# 
# La matriz de covarianza mide cómo los retornos de los diferentes valores se mueven juntos. Esta matriz es esencial para calcular la volatilidad de la cartera.

# In[78]:


matriz_covarianza = np.zeros((len(valores), len(valores)))
for i, (_, datos_i) in enumerate(valores.items()):
    for j, (_, datos_j) in enumerate(valores.items()):
        matriz_covarianza[i][j] = datos_i['Close'].pct_change().cov(datos_j['Close'].pct_change()) * 252


# Paso 3: Optimización de la Cartera usando Gradient Boosting
# 
# Generamos 10,000 combinaciones de pesos al azar para los valores en la cartera y calculamos el retorno esperado, la volatilidad y el ratio Sharpe para cada combinación. 
# 
# Luego, entrenamos un modelo de Gradient Boosting para predecir el ratio Sharpe en función de los pesos y utilizamos el modelo para encontrar los pesos que maximizan y minimizan el ratio Sharpe.

# In[79]:


X = np.random.random((10000, len(valores)))
X /= np.sum(X, axis=1)[:, np.newaxis]


# In[80]:


retornos = np.array([retorno for _, retorno in retorno_volatilidad.values()])
volatilidades = np.sqrt(np.dot(X, np.dot(matriz_covarianza, X.T)).diagonal())


# In[81]:


ratios_sharpe = retornos @ X.T / volatilidades


# In[82]:


gb = GradientBoostingRegressor(n_estimators=100, max_depth=3)
gb.fit(X, ratios_sharpe)


# In[83]:


# Encontrar los pesos que maximizan el ratio Sharpe
pesos_optimales = X[np.argmax(gb.predict(X))]
retorno_optimo = retornos @ pesos_optimales
volatilidad_optima = np.sqrt(pesos_optimales.T @ np.dot(matriz_covarianza, pesos_optimales))
sharpe_max = retorno_optimo / volatilidad_optima


# In[84]:


# Encontrar los pesos que minimizan el ratio Sharpe
pesos_minimos = X[np.argmin(gb.predict(X))]
retorno_minimo = retornos @ pesos_minimos
volatilidad_minima = np.sqrt(pesos_minimos.T @ np.dot(matriz_covarianza, pesos_minimos))
sharpe_min = retorno_minimo / volatilidad_minima


# Paso 4: Mostrar Resultados
# 
# Mostramos los resultados de la optimización, incluyendo los pesos de los valores en la cartera, el retorno, la volatilidad y el ratio Sharpe.

# In[85]:


print(f'Ratio Sharpe Máximo: {sharpe_max}')
print(f'Retorno de la Cartera (Sharpe Máximo): {retorno_optimo}')
print(f'Volatilidad de la Cartera (Sharpe Máximo): {volatilidad_optima}')
print('Pesos de la Cartera (Sharpe Máximo):')
for nombre, peso in zip(valores.keys(), pesos_optimales):
    print(f'{nombre}: {peso:.2f}%')


# In[86]:


print(f'Ratio Sharpe Mínimo: {sharpe_min}')
print(f'Retorno de la Cartera (Sharpe Mínimo): {retorno_minimo}')
print(f'Volatilidad de la Cartera (Sharpe Mínimo): {volatilidad_minima}')
print('Pesos de la Cartera (Sharpe Mínimo):')
for nombre, peso in zip(valores.keys(), pesos_minimos):
    print(f'{nombre}: {peso:.2f}%')


# Paso 5: Frontera Eficiente de Markowitz
# 
# Calculamos la frontera eficiente de Markowitz, que representa las carteras que ofrecen el máximo retorno para un nivel dado de riesgo. Utilizamos optimización numérica para encontrar estas carteras.

# In[87]:


frontera_volatilidad = np.linspace(0, 0.3, 100)
frontera_retorno = []
for volatilidad in frontera_volatilidad:
    restricciones_frontera = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1},
                              {'type': 'eq', 'fun': lambda pesos: np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos))) - volatilidad})
    resultados_frontera = minimize(lambda pesos: -np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos), pesos_optimales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones_frontera)
    retorno = np.dot([retorno for _, retorno in retorno_volatilidad.values()], resultados_frontera.x)
    frontera_retorno.append(retorno)


# Paso 6: Graficar la Frontera Eficiente y los Resultados
# 
# Graficamos la frontera eficiente y los puntos de los valores individuales.

# In[88]:


fig = go.Figure()

# Añadir puntos de valores individuales
for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    fig.add_trace(go.Scatter(x=[volatilidad], y=[retorno], mode='markers', name=nombre))

# Añadir la frontera eficiente de Markowitz
fig.add_trace(go.Scatter(x=frontera_volatilidad, y=frontera_retorno, mode='lines', name='Frontera eficiente de Markowitz', line=dict(color='red', dash='dash')))

fig.update_layout(
    title='Portafolios Simulados y Frontera Eficiente de Markowitz',
    xaxis_title='Volatilidad',
    yaxis_title='Retorno esperado',
    showlegend=True,
    legend=dict(
        orientation="v", 
        yanchor="top", 
        y=1,
        xanchor="left",
        x=1.05
    ),
    width=1000,  
    height=600  
)

fig.show()


# Paso 7: Crear Gráfico Interactivo Circular con los Pesos de la Cartera Óptima
# 
# Finalmente, creamos un gráfico circular para visualizar los pesos de la cartera óptima (con el ratio Sharpe máximo).

# In[89]:


pesos_labels_max = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_optimales * 100)]
fig_pie_max = go.Figure(data=[go.Pie(labels=pesos_labels_max, values=pesos_optimales * 100, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
fig_pie_max.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
fig_pie_max.update_layout(title='Pesos de la Cartera Óptima (Sharpe Máximo)', showlegend=False)
fig_pie_max.show()


# ### Resultados finales modelo 3:

# In[90]:


retorno_cartera_modelo_3 = retorno_optimo
volatilidad_cartera_modelo_3 = volatilidad_optima
sharpe_cartera_modelo_3 = sharpe_max


# In[91]:


retorno_cartera_modelo_3


# In[92]:


volatilidad_cartera_modelo_3


# In[93]:


sharpe_cartera_modelo_3


# In[94]:


pesos_cartera_optima_modelo_3 = {nombre: peso for nombre, peso in zip(valores.keys(), pesos_optimales)}
pesos_cartera_optima_modelo_3


# ## 6.2.- Modelo Supervisado ML con XGBoost (Modelo 4)

# In[95]:


import xgboost as xgb


# In[96]:


ACS_dataset = pd.read_csv('../data/ACS_dataset.csv')
SAN_dataset = pd.read_csv('../data/SAN_dataset.csv')
BBVA_dataset = pd.read_csv('../data/BBVA_dataset.csv')
REP_dataset = pd.read_csv('../data/REP_dataset.csv')
IBE_dataset = pd.read_csv('../data/IBE_dataset.csv')
NIFTY50_dataset = pd.read_csv('../data/NIFTY50_dataset.csv')
ASX200_dataset = pd.read_csv('../data/ASX200_dataset.csv')
SP500_dataset = pd.read_csv('../data/SP500_dataset.csv')
Nasdaq_dataset = pd.read_csv('../data/Nasdaq_dataset.csv')
SPCanada_dataset = pd.read_csv('../data/SPCanada_dataset.csv')


# Paso 1: Calcular Retorno y Volatilidad para Cada Valor
# 
# Primero, calculamos el retorno esperado anualizado y la volatilidad anualizada para cada uno de los 10 valores.

# In[97]:


# Función calcular el retorno y la volatilidad
def calcular_retorno_volatilidad(datos):
    retorno = datos['Close'].pct_change().mean() * 252
    volatilidad = datos['Close'].pct_change().std() * np.sqrt(252)
    return retorno, volatilidad


# In[98]:


valores = {
    'ACS': ACS_dataset,
    'SAN': SAN_dataset,
    'BBVA': BBVA_dataset,
    'REP': REP_dataset,
    'IBE': IBE_dataset,
    'NIFTY50': NIFTY50_dataset,
    'ASX200': ASX200_dataset,
    'SP500': SP500_dataset,
    'Nasdaq': Nasdaq_dataset,
    'SPCanada': SPCanada_dataset
}


# In[99]:


retorno_volatilidad = {nombre: calcular_retorno_volatilidad(datos) for nombre, datos in valores.items()}


# In[100]:


for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    print(f'{nombre}: Retorno={retorno}, Volatilidad={volatilidad}')


# Paso 2: Crear la Matriz de Covarianza
# 
# La matriz de covarianza mide cómo los retornos de los diferentes valores se mueven juntos. Esta matriz es esencial para calcular la volatilidad de la cartera.

# In[101]:


matriz_covarianza = np.zeros((len(valores), len(valores)))
for i, (_, datos_i) in enumerate(valores.items()):
    for j, (_, datos_j) in enumerate(valores.items()):
        matriz_covarianza[i][j] = datos_i['Close'].pct_change().cov(datos_j['Close'].pct_change()) * 252


# Paso 3: Optimización de la Cartera usando XGBoost
# 
# Generamos 10,000 combinaciones de pesos al azar para los valores en la cartera y calculamos el retorno esperado, la volatilidad y el ratio Sharpe para cada combinación. 
# 
# Luego, entrenamos un modelo de XGBoost para predecir el ratio Sharpe, optimizando posteriormente la cartera.

# In[102]:


num_simulaciones = 10000
resultados = np.zeros((num_simulaciones, len(valores) + 3))

for i in range(num_simulaciones):
    pesos = np.random.random(len(valores))
    pesos /= np.sum(pesos)
    
    retorno_esperado = np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos)
    volatilidad_esperada = np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))
    ratio_sharpe = retorno_esperado / volatilidad_esperada
    
    resultados[i, :len(valores)] = pesos
    resultados[i, len(valores)] = retorno_esperado
    resultados[i, len(valores) + 1] = volatilidad_esperada
    resultados[i, len(valores) + 2] = ratio_sharpe


# In[103]:


X = resultados[:, :len(valores)]
y = resultados[:, len(valores) + 2]


# In[104]:


dtrain = xgb.DMatrix(X, label=y)
params = {'max_depth': 3, 'eta': 0.1, 'objective': 'reg:squarederror'}
num_round = 100
bst = xgb.train(params, dtrain, num_round)


# In[105]:


dtest = xgb.DMatrix(X)
predicciones = bst.predict(dtest)


# In[106]:


pesos_optimales = X[np.argmax(predicciones)]
retorno_optimo = np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos_optimales)
volatilidad_optima = np.sqrt(np.dot(pesos_optimales.T, np.dot(matriz_covarianza, pesos_optimales)))
sharpe_max = retorno_optimo / volatilidad_optima


# In[107]:


pesos_minimos = X[np.argmin(predicciones)]
retorno_minimo = np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos_minimos)
volatilidad_minima = np.sqrt(np.dot(pesos_minimos.T, np.dot(matriz_covarianza, pesos_minimos)))
sharpe_min = retorno_minimo / volatilidad_minima


# Paso 4: Mostrar Resultados
# 
# Mostramos los resultados de la optimización, incluyendo los pesos de los valores en la cartera, el retorno, la volatilidad y el ratio Sharpe.

# In[108]:


print(f'Ratio Sharpe Máximo: {sharpe_max}')
print(f'Retorno de la Cartera (Sharpe Máximo): {retorno_optimo}')
print(f'Volatilidad de la Cartera (Sharpe Máximo): {volatilidad_optima}')
print('Pesos de la Cartera (Sharpe Máximo):')
for nombre, peso in zip(valores.keys(), pesos_optimales):
    print(f'{nombre}: {peso:.2f}%')


# In[109]:


print(f'Ratio Sharpe Mínimo: {sharpe_min}')
print(f'Retorno de la Cartera (Sharpe Mínimo): {retorno_minimo}')
print(f'Volatilidad de la Cartera (Sharpe Mínimo): {volatilidad_minima}')
print('Pesos de la Cartera (Sharpe Mínimo):')
for nombre, peso in zip(valores.keys(), pesos_minimos):
    print(f'{nombre}: {peso:.2f}%')


# Paso 5: Frontera Eficiente de Markowitz
# 
# Calculamos la frontera eficiente de Markowitz, que representa las carteras que ofrecen el máximo retorno para un nivel dado de riesgo. Utilizamos optimización numérica para encontrar estas carteras.

# In[110]:


frontera_volatilidad = np.linspace(0, 0.3, 100)
frontera_retorno = []
for volatilidad in frontera_volatilidad:
    restricciones_frontera = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1},
                              {'type': 'eq', 'fun': lambda pesos: np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos))) - volatilidad})
    resultados_frontera = minimize(lambda pesos: -np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos), pesos_optimales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones_frontera)
    retorno = np.dot([retorno for _, retorno in retorno_volatilidad.values()], resultados_frontera.x)
    frontera_retorno.append(retorno)


# Paso 6: Graficar la Frontera Eficiente y los Resultados
# 
# Graficamos la frontera eficiente y los puntos de los valores individuales.

# In[111]:


fig = go.Figure()

# Añadir puntos de valores individuales
for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    fig.add_trace(go.Scatter(x=[volatilidad], y=[retorno], mode='markers', name=nombre))

# Añadir la frontera eficiente de Markowitz
fig.add_trace(go.Scatter(x=frontera_volatilidad, y=frontera_retorno, mode='lines', name='Frontera eficiente de Markowitz', line=dict(color='red', dash='dash')))

fig.update_layout(
    title='Portafolios Simulados y Frontera Eficiente de Markowitz',
    xaxis_title='Volatilidad',
    yaxis_title='Retorno esperado',
    showlegend=True,
    legend=dict(
        orientation="v",  
        yanchor="top", 
        y=1,
        xanchor="left",
        x=1.05
    ),
    width=1000,  
    height=600  
)

fig.show()


# Paso 7: Crear Gráfico Interactivo Circular con los Pesos de la Cartera Óptima
# 
# Finalmente, creamos un gráfico circular para visualizar los pesos de la cartera óptima (con el ratio Sharpe máximo).

# In[112]:


pesos_labels_max = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_optimales * 100)]
fig_pie_max = go.Figure(data=[go.Pie(labels=pesos_labels_max, values=pesos_optimales * 100, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
fig_pie_max.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
fig_pie_max.update_layout(title='Pesos de la Cartera Óptima (Sharpe Máximo)', showlegend=False)
fig_pie_max.show()


# ### Resultados finales modelo 4:

# In[113]:


retorno_cartera_modelo_4 = retorno_optimo
volatilidad_cartera_modelo_4 = volatilidad_optima
sharpe_cartera_modelo_4 = sharpe_max


# In[114]:


retorno_cartera_modelo_4


# In[115]:


volatilidad_cartera_modelo_4


# In[116]:


sharpe_cartera_modelo_4


# In[117]:


pesos_cartera_optima_modelo_4 = {nombre: peso for nombre, peso in zip(valores.keys(), pesos_optimales)}
pesos_cartera_optima_modelo_4


# ## 6.3.- Modelo No Supervisado PCA (Principal Component Analysis) (Modelo 5)

# In[118]:


from sklearn.decomposition import PCA


# In[119]:


ACS_dataset = pd.read_csv('../data/ACS_dataset.csv')
SAN_dataset = pd.read_csv('../data/SAN_dataset.csv')
BBVA_dataset = pd.read_csv('../data/BBVA_dataset.csv')
REP_dataset = pd.read_csv('../data/REP_dataset.csv')
IBE_dataset = pd.read_csv('../data/IBE_dataset.csv')
NIFTY50_dataset = pd.read_csv('../data/NIFTY50_dataset.csv')
ASX200_dataset = pd.read_csv('../data/ASX200_dataset.csv')
SP500_dataset = pd.read_csv('../data/SP500_dataset.csv')
Nasdaq_dataset = pd.read_csv('../data/Nasdaq_dataset.csv')
SPCanada_dataset = pd.read_csv('../data/SPCanada_dataset.csv')


# Paso 1: Calcular Retorno y Volatilidad para Cada Valor
# 
# Primero, necesitamos calcular el retorno esperado anualizado y la volatilidad anualizada para cada uno de los 10 valores.

# In[120]:


# Función para calcular el retorno y la volatilidad
def calcular_retorno_volatilidad(datos):
    retorno = datos['Close'].pct_change().mean() * 252
    volatilidad = datos['Close'].pct_change().std() * np.sqrt(252)
    return retorno, volatilidad


# In[121]:


valores = {
    'ACS': ACS_dataset,
    'SAN': SAN_dataset,
    'BBVA': BBVA_dataset,
    'REP': REP_dataset,
    'IBE': IBE_dataset,
    'NIFTY50': NIFTY50_dataset,
    'ASX200': ASX200_dataset,
    'SP500': SP500_dataset,
    'Nasdaq': Nasdaq_dataset,
    'SPCanada': SPCanada_dataset
}


# In[122]:


retorno_volatilidad = {nombre: calcular_retorno_volatilidad(datos) for nombre, datos in valores.items()}


# In[123]:


for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    print(f'{nombre}: Retorno={retorno}, Volatilidad={volatilidad}')


# Paso 2: Crear la Matriz de Covarianza
# 
# La matriz de covarianza se utiliza para medir cómo los retornos de los valores se mueven juntos. Esta matriz es esencial para calcular la volatilidad de la cartera.

# In[124]:


matriz_covarianza = np.zeros((len(valores), len(valores)))
for i, (_, datos_i) in enumerate(valores.items()):
    for j, (_, datos_j) in enumerate(valores.items()):
        matriz_covarianza[i][j] = datos_i['Close'].pct_change().cov(datos_j['Close'].pct_change()) * 252


# Paso 3: Aplicar PCA
# 
# Utilizamos PCA para reducir la dimensionalidad y encontrar las componentes principales que expliquen la mayor parte de la variabilidad en los retornos.
# 
#  Alineamos los índices de los DataFrames para asegurar que todas las series temporales tengan la misma longitud.

# In[125]:


# Alinear los índices de los DataFrames para asegurar que todas las series temporales tengan la misma longitud
retornos_diarios_df = pd.concat([datos['Close'].pct_change().dropna() for _, datos in valores.items()], axis=1, join='inner')
retornos_diarios_df.columns = valores.keys()


# In[126]:


# Convertir DataFrame a matriz numpy
retornos_diarios = retornos_diarios_df.values


# In[127]:


# Aplicar PCA
pca = PCA()
pca.fit(retornos_diarios)
componentes_principales = pca.transform(retornos_diarios)


# In[128]:


# Reconstruir los datos utilizando solo las componentes principales más significativas
num_componentes = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.95) + 1
retornos_reconstruidos = np.dot(componentes_principales[:, :num_componentes], pca.components_[:num_componentes, :])
retornos_reconstruidos += pca.mean_


# Paso 4: Optimización de la Cartera usando las Componentes Principales
# 
# Utilizamos las componentes principales para optimizar la cartera.

# In[129]:


# Calcular el retorno y la volatilidad de la cartera utilizando las componentes principales
retornos_medios = retornos_reconstruidos.mean(axis=0) * 252
volatilidades = retornos_reconstruidos.std(axis=0) * np.sqrt(252)
matriz_covarianza_pca = np.cov(retornos_reconstruidos.T) * 252


# In[130]:


# Función objetivo para minimizar la volatilidad de la cartera
def objetivo(pesos):
    return np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza_pca, pesos)))


# In[131]:


# Restricciones para los pesos de la cartera
restricciones = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1})


# In[132]:


# Optimización
pesos_iniciales = np.ones(len(valores)) / len(valores)
resultados = minimize(objetivo, pesos_iniciales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones)


# In[133]:


# Calcular retorno, volatilidad y ratio Sharpe de la cartera óptima
retorno_cartera = np.dot(retornos_medios, resultados.x)
volatilidad_cartera = np.sqrt(np.dot(resultados.x.T, np.dot(matriz_covarianza_pca, resultados.x)))
sharpe_cartera = retorno_cartera / volatilidad_cartera


# In[134]:


# Encontrar los pesos que maximizan el ratio Sharpe
pesos_sharpe_max = resultados.x
retorno_sharpe_max = retorno_cartera
volatilidad_sharpe_max = volatilidad_cartera
sharpe_max = sharpe_cartera


# In[135]:


# Encontrar los pesos que minimizan el ratio Sharpe (maximizar la penalización negativa)
resultados_min = minimize(lambda pesos: -np.dot(retornos_medios, pesos) / np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza_pca, pesos))), pesos_iniciales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones)
pesos_sharpe_min = resultados_min.x
retorno_sharpe_min = np.dot(retornos_medios, pesos_sharpe_min)
volatilidad_sharpe_min = np.sqrt(np.dot(pesos_sharpe_min.T, np.dot(matriz_covarianza_pca, pesos_sharpe_min)))
sharpe_min = retorno_sharpe_min / volatilidad_sharpe_min


# Paso 5: Mostrar Resultados
# 
# Mostramos los resultados de la optimización, incluyendo los pesos de los valores en la cartera, el retorno, la volatilidad y el ratio Sharpe.

# In[136]:


print(f'Ratio Sharpe Máximo: {sharpe_max}')
print(f'Retorno de la Cartera (Sharpe Máximo): {retorno_sharpe_max}')
print(f'Volatilidad de la Cartera (Sharpe Máximo): {volatilidad_sharpe_max}')
print('Pesos de la Cartera (Sharpe Máximo):')
for nombre, peso in zip(valores.keys(), pesos_sharpe_max):
    print(f'{nombre}: {peso:.2f}%')


# In[137]:


print(f'Ratio Sharpe Mínimo: {sharpe_min}')
print(f'Retorno de la Cartera (Sharpe Mínimo): {retorno_sharpe_min}')
print(f'Volatilidad de la Cartera (Sharpe Mínimo): {volatilidad_sharpe_min}')
print('Pesos de la Cartera (Sharpe Mínimo):')
for nombre, peso in zip(valores.keys(), pesos_sharpe_min):
    print(f'{nombre}: {peso:.2f}%')


# Paso 6: Frontera Eficiente de Markowitz
# 
# Calculamos la frontera eficiente de Markowitz, que representa las carteras que ofrecen el máximo retorno para un nivel dado de riesgo.

# In[138]:


frontera_volatilidad = np.linspace(0, 0.3, 100)
frontera_retorno = []
for volatilidad in frontera_volatilidad:
    restricciones_frontera = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1},
                              {'type': 'eq', 'fun': lambda pesos: np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza_pca, pesos))) - volatilidad})
    resultados_frontera = minimize(lambda pesos: -np.dot(retornos_medios, pesos), pesos_iniciales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones_frontera)
    retorno = np.dot(retornos_medios, resultados_frontera.x)
    frontera_retorno.append(retorno)


# Paso 7: Graficar la Frontera Eficiente y los Resultados
# 
# Graficamos la frontera eficiente y los puntos de los valores individuales.

# In[139]:


fig = go.Figure()

# Añadir puntos de valores individuales
for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    fig.add_trace(go.Scatter(x=[volatilidad], y=[retorno], mode='markers', name=nombre))

# Añadir la frontera eficiente de Markowitz
fig.add_trace(go.Scatter(x=frontera_volatilidad, y=frontera_retorno, mode='lines', name='Frontera eficiente de Markowitz', line=dict(color='red', dash='dash')))

fig.update_layout(
    title='Portafolios Simulados y Frontera Eficiente de Markowitz',
    xaxis_title='Volatilidad',
    yaxis_title='Retorno',
    showlegend=True
)

fig.show()


# Paso 8: Crear Gráfico Interactivo Circular con los Pesos de la Cartera Óptima
# 
# Finalmente, creamos un gráfico circular para visualizar los pesos de la cartera óptima (con el ratio Sharpe máximo).

# In[140]:


pesos_labels_max = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_sharpe_max * 100)]
fig_pie_max = go.Figure(data=[go.Pie(labels=pesos_labels_max, values=pesos_sharpe_max * 100, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
fig_pie_max.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
fig_pie_max.update_layout(title='Pesos de la Cartera Óptima (Sharpe Máximo)', showlegend=False)
fig_pie_max.show()


# ### Resultados finales modelo 5:

# In[141]:


retorno_cartera_modelo_5 = retorno_sharpe_max
volatilidad_cartera_modelo_5 = volatilidad_sharpe_max
sharpe_cartera_modelo_5 = sharpe_max


# In[142]:


retorno_cartera_modelo_5


# In[143]:


volatilidad_cartera_modelo_5


# In[144]:


sharpe_cartera_modelo_5


# In[145]:


pesos_cartera_optima_modelo_5 = {nombre: peso for nombre, peso in zip(valores.keys(), pesos_sharpe_max)}
pesos_cartera_optima_modelo_5


# # 7.- Comparacion de resultados finales de los 5 modelos

# ## 7.1.- Comparacion de retorno, volatilidad y sharpe de las 5 carteras

# In[146]:


resultados_modelos = {
    'Modelo 1': {
        'retorno': retorno_cartera_modelo_1,
        'volatilidad': volatilidad_cartera_modelo_1,
        'sharpe': sharpe_cartera_modelo_1
    },
    'Modelo 2': {
        'retorno': retorno_cartera_modelo_2,
        'volatilidad': volatilidad_cartera_modelo_2,
        'sharpe': sharpe_cartera_modelo_2
    },
    'Modelo 3': {
        'retorno': retorno_cartera_modelo_3,
        'volatilidad': volatilidad_cartera_modelo_3,
        'sharpe': sharpe_cartera_modelo_3
    },
    'Modelo 4': {
        'retorno': retorno_cartera_modelo_4,
        'volatilidad': volatilidad_cartera_modelo_4,
        'sharpe': sharpe_cartera_modelo_4
    },
    'Modelo 5': {
        'retorno': retorno_cartera_modelo_5,
        'volatilidad': volatilidad_cartera_modelo_5,
        'sharpe': sharpe_cartera_modelo_5
    }
}


# Crear la Tabla de Comparación

# In[147]:


df_comparacion = pd.DataFrame({
    'Modelo': resultados_modelos.keys(),
    'Retorno': [resultados_modelos[modelo]['retorno'] for modelo in resultados_modelos],
    'Volatilidad': [resultados_modelos[modelo]['volatilidad'] for modelo in resultados_modelos],
    'Ratio Sharpe': [resultados_modelos[modelo]['sharpe'] for modelo in resultados_modelos]
})


# In[148]:


df_comparacion.set_index('Modelo', inplace=True)


# In[149]:


print(df_comparacion)


# Visualización de resultados

# In[151]:


trace_retorno = go.Bar(
    x=df_comparacion.index,
    y=df_comparacion['Retorno'],
    name='Retorno',
    marker=dict(color='skyblue')
)

trace_volatilidad = go.Bar(
    x=df_comparacion.index,
    y=df_comparacion['Volatilidad'],
    name='Volatilidad',
    marker=dict(color='lightcoral')
)

trace_sharpe = go.Bar(
    x=df_comparacion.index,
    y=df_comparacion['Ratio Sharpe'],
    name='Ratio Sharpe',
    marker=dict(color='lightgreen')
)

fig = go.Figure(data=[trace_retorno, trace_volatilidad, trace_sharpe])

fig.update_layout(
    title='Comparación de Retorno, Volatilidad y Ratio Sharpe entre Modelos',
    xaxis_title='Modelos',
    yaxis_title='Valores',
    barmode='group',
    template='plotly_white'
)

fig.show()


# Para entender cuál es el mejor modelo, consideramos las tres métricas: retorno, volatilidad y ratio Sharpe. El ratio Sharpe es particularmente importante porque mide el rendimiento ajustado por riesgo de una cartera.
# 
# Siendo:
# 
# - Retorno: una medida del beneficio esperado de la cartera
# - Volatilidad: una medida del riesgo o la variabilidad de los retornos de la cartera
# - El ratio Sharpe: mide el rendimiento ajustado por riesgo. Un ratio Sharpe más alto indica un mejor rendimiento ajustado por riesgo

# Interpretación de los Resultados
# 
# 1) Modelo 1: Tiene el menor riesgo (volatilidad), pero no el mayor retorno ni el mayor ratio Sharpe.
# 
# 2) Modelo 2: Ofrece el mayor retorno pero con un riesgo relativamente más alto y un ratio Sharpe menor que el Modelo 3 y 4.
# 
# 3) Modelo 3: Tiene el mayor ratio Sharpe, lo que indica el mejor rendimiento ajustado por riesgo. Aunque no tiene el mayor retorno, combina un buen retorno con una volatilidad moderada, resultando en el mejor ratio Sharpe.
# 
# 4) Modelo 4: Tiene un ratio Sharpe muy cercano al del Modelo 3, pero ligeramente inferior.
# 
# 5) Modelo 5: Tiene el menor retorno y el menor ratio Sharpe, lo que indica un rendimiento inferior ajustado por riesgo en comparación con los otros modelos.

# Conclusión
# 
# El Modelo 3 es el mejor modelo según los resultados, ya que ofrece el mejor rendimiento ajustado por riesgo, como se evidencia por su ratio Sharpe más alto. Aunque el Modelo 2 tiene el mayor retorno, su mayor volatilidad reduce su ratio Sharpe en comparación con el Modelo 3. Por lo tanto, el Modelo 3 proporciona una combinación óptima de retorno y riesgo.

# ## 7.2.- Comparacion de peso de carteras

# Primero recogemos los datos. Puesto que los modelos 2, 3, 4 y 5 mostraban los resultados de los porcentajes, en formato decimal (20 % = 0.20), van a tener que multiplicarse sus resultados por 100. Con el fin de iguar los resultados con los del modelo 1.

# In[159]:


pesos_modelos = {
    'Modelo 1': pesos_cartera_optima_modelo_1,
    'Modelo 2': {k: v * 100 for k, v in pesos_cartera_optima_modelo_2.items()},
    'Modelo 3': {k: v * 100 for k, v in pesos_cartera_optima_modelo_3.items()},
    'Modelo 4': {k: v * 100 for k, v in pesos_cartera_optima_modelo_4.items()},
    'Modelo 5': {k: v * 100 for k, v in pesos_cartera_optima_modelo_5.items()}
}


# Crear la Tabla de Comparación de Pesos
# 
# Convertimos los datos de los pesos en un DataFrame para facilitar su manipulación y visualización.

# In[160]:


# DataFrame para la comparación de pesos
df_pesos = pd.DataFrame(pesos_modelos)
df_pesos.index.name = 'Activos'
df_pesos.reset_index(inplace=True)
print(df_pesos)


# In[161]:


trazas = []
for modelo in df_pesos.columns[1:]:
    traza = go.Bar(
        x=df_pesos['Activos'],
        y=df_pesos[modelo],
        name=modelo
    )
    trazas.append(traza)

fig = go.Figure(data=trazas)

fig.update_layout(
    title='Comparación de Pesos de Carteras entre Modelos',
    xaxis_title='Activos',
    yaxis_title='Peso (%)',
    barmode='group',
    template='plotly_white'
)

fig.show()


# Análisis y explicación de resultados
# 
# 1) Modelo 1: Este modelo ha asignado la mayor parte de la inversión a los índices ASX200 y NIFTY50, con una diversificación significativa en SPCanada. La inversión en bancos (SAN, BBVA) y Nasdaq es nula, sugiriendo una preferencia por índices de mercados más estables.
# 
# 2) Modelo 2: Similar al Modelo 1, este modelo también favorece NIFTY50 y ASX200, pero incluye una mayor diversificación en REP (8.93%) y BBVA (6.17%). La exposición a bancos y la energía es mayor, con una reducción en la inversión en SP500.
# 
# 3) Modelo 3: Este modelo presenta una diversificación más equilibrada, incluyendo una inversión notable en Nasdaq (9.69%) y REP (6.02%). La menor inversión en ACS y SP500 sugiere una estrategia de minimización de riesgo en estos sectores.
# 
# 4) Modelo 4: Este modelo destaca por su alta diversificación y un peso significativo en Nasdaq (12.45%). La inversión en bancos es baja, indicando una posible estrategia de evitar volatilidades asociadas a este sector.
# 
# 5) Modelo 5: Al igual que los otros modelos, NIFTY50 y ASX200 son los favoritos, pero este modelo muestra una estrategia más conservadora al excluir completamente a BBVA y Nasdaq. La inversión en IBE es elevada (11.68%), indicando una preferencia por este sector.

# Conclusiones
# 
# Cada modelo muestra una estrategia de inversión diferente, pero hay algunas tendencias comunes:
# 
# - Favoritismo por NIFTY50 y ASX200: Todos los modelos asignan una gran parte de la inversión a estos índices, indicando una confianza general en estos mercados.
# - Diversificación en SPCanada: También es un activo popular en todas las carteras.
# - Variabilidad en Activos Específicos: La inversión en bancos y Nasdaq varía significativamente entre los modelos, sugiriendo diferentes enfoques hacia la volatilidad y el riesgo.
# 
# El Modelo 3 parece tener el mejor balance de diversificación y riesgo ajustado, como se indicó anteriormente en el análisis del ratio Sharpe. Sin embargo, la elección del modelo final puede depender de las preferencias de riesgo y la estrategia de inversión del inversor.
