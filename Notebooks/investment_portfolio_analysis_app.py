
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize

# Cargar datos
@st.cache
def load_data():
    df = pd.read_csv('path_to_your_file.csv')  # Adjust the file path accordingly
    return df

def main():
    st.title('Investment Portfolio Analysis')

    # Cargar datos
    df = load_data()
    st.write("### Data Head")
    st.write(df.head())

    # Mostrar descripciones estadísticas
    st.write("### Data Description")
    st.write(df.describe())

    # Correlación
    st.write("### Correlation Matrix")
    corr_matrix = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, ax=ax)
    st.pyplot(fig)

    # Distribución de una columna específica
    st.write("### Distribution of a specific column")
    fig, ax = plt.subplots()
    sns.histplot(df['specific_column'], kde=True, ax=ax)  # Replace 'specific_column' with the actual column name
    st.pyplot(fig)

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

ACS_dataset = pd.read_csv('../data/ACS_dataset.csv')
ACS_dataset

NIFTY50_dataset = pd.read_csv('../data/NIFTY50_dataset.csv')
NIFTY50_dataset

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

company_names = [
    'ANA', 'ANE', 'ACS', 'ACX', 'AENA', 'AMS', 'MTS', 'SAB', 'SAN', 'BKT', 
    'BBVA', 'CABK', 'CLNX', 'COL', 'ENG', 'ELE', 'FER', 'FDR', 'GRF', 'IAG', 
    'IBE', 'IDR', 'ITX', 'LOG', 'MAP', 'MEL', 'MRL', 'NTGY', 'RED', 'REP', 
    'ROVI', 'SCYR', 'SLR', 'TEF', 'UNI'
]

dataframes = []
for file, name in zip(files, company_names):
    df = pd.read_csv(f'../data/{file}')
    df['Company'] = name
    dataframes.append(df)

all_data = pd.concat(dataframes, ignore_index=True)

all_data['Date'] = pd.to_datetime(all_data['Date'])

fig = px.line(
    all_data,
    x='Date',
    y='Close',
    color='Company',
    title='Precio de cierre (Close) en el período 01/01/2018 - 20/05/2024',
    labels={'Close': 'Precio de cierre (EUR)', 'Date': 'Fecha'}
)

fig.show()

index_files = [
    'NIFTY50_dataset.csv', 'ASX200_dataset.csv', 'SP500_dataset.csv', 'DowJones_dataset.csv',
    'Nasdaq_dataset.csv', 'SPCanada_dataset.csv', 'DAX40_dataset.csv', 'FTSE100_dataset.csv',
    'Euro50_dataset.csv', 'IBEX35_dataset.csv'
]

index_names = [
    'NIFTY50', 'ASX200', 'SP500', 'DowJones', 'Nasdaq',
    'SPCanada', 'DAX40', 'FTSE100', 'Euro50', 'IBEX35'
]

index_dataframes = []
for file, name in zip(index_files, index_names):
    df = pd.read_csv(f'../data/{file}')
    df['Index'] = name
    index_dataframes.append(df)

all_index_data = pd.concat(index_dataframes, ignore_index=True)

all_index_data['Date'] = pd.to_datetime(all_index_data['Date'])

fig = px.line(
    all_index_data,
    x='Date',
    y='Close',
    color='Index',
    title='Valor de cierre de los 10 índices entre 01/01/2018 y 20/05/2024',
    labels={'Close': 'Valor de cierre (puntos)', 'Date': 'Fechas'}
)

fig.show()

# Lista para almacenar las rentabilidades
returns = []

# Calcular la rentabilidad para cada empresa
for file, name in zip(files, company_names):
    df = pd.read_csv(f'../data/{file}')
    initial_close = df['Close'].iloc[0]
    final_close = df['Close'].iloc[-1]
    return_percentage = ((final_close - initial_close) / initial_close) * 100
    returns.append({'Company': name, 'Return': return_percentage})

returns_df = pd.DataFrame(returns)

# Ordenar el DataFrame por la columna 'Return' en orden descendente
returns_df = returns_df.sort_values(by='Return', ascending=False)

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

returns = []

for file, name in zip(index_files, index_names):
    df = pd.read_csv(f'../data/{file}')
    initial_close = df['Close'].iloc[0]
    final_close = df['Close'].iloc[-1]
    return_percentage = ((final_close - initial_close) / initial_close) * 100
    returns.append({'Index': name, 'Return': return_percentage})

returns_df = pd.DataFrame(returns)

returns_df = returns_df.sort_values(by='Return', ascending=False)

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


from scipy.optimize import minimize

def calcular_retorno_volatilidad(datos):
    retorno = datos['Close'].pct_change().mean() * 252
    volatilidad = datos['Close'].pct_change().std() * np.sqrt(252)
    return retorno, volatilidad

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

retorno_volatilidad = {}
for nombre, datos in valores.items():
    retorno_volatilidad[nombre] = calcular_retorno_volatilidad(datos)

for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    print(f'{nombre}: Retorno={retorno}, Volatilidad={volatilidad}')

matriz_covarianza = np.zeros((len(valores), len(valores)))
for i, (_, datos_i) in enumerate(valores.items()):
    for j, (_, datos_j) in enumerate(valores.items()):
        matriz_covarianza[i][j] = datos_i['Close'].pct_change().cov(datos_j['Close'].pct_change()) * 252

def objetivo(pesos):
    return np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))

restricciones = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1})

pesos_iniciales = np.ones(len(valores)) / len(valores)
resultados = minimize(objetivo, pesos_iniciales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones)

retorno_cartera = np.dot([retorno for _, retorno in retorno_volatilidad.values()], resultados.x)
volatilidad_cartera = np.sqrt(np.dot(resultados.x.T, np.dot(matriz_covarianza, resultados.x)))
sharpe_cartera = retorno_cartera / volatilidad_cartera

print(f'Retorno de la cartera: {retorno_cartera}')
print(f'Volatilidad de la cartera: {volatilidad_cartera}')
print(f'Ratio Sharpe de la cartera: {sharpe_cartera}')

pesos_cartera_optima = resultados.x * 100
print('Pesos de la cartera óptima:')
for nombre, peso in zip(valores.keys(), pesos_cartera_optima):
    print(f'{nombre}: {peso:.2f}%')

frontera_volatilidad = np.linspace(0, 0.3, 100)
frontera_retorno = []
for volatilidad in frontera_volatilidad:
    restricciones_frontera = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1},
                              {'type': 'eq', 'fun': lambda pesos: np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos))) - volatilidad})
    resultados_frontera = minimize(lambda pesos: -np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos), pesos_iniciales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones_frontera)
    retorno = np.dot([retorno for _, retorno in retorno_volatilidad.values()], resultados_frontera.x)
    frontera_retorno.append(retorno)

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

pesos_cartera_optima_pct = resultados.x * 100
pesos_labels = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_cartera_optima_pct)]
fig_pie = go.Figure(data=[go.Pie(labels=pesos_labels, values=pesos_cartera_optima_pct, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
fig_pie.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
fig_pie.update_layout(title='Pesos de la cartera óptima', showlegend=False)
fig_pie.show()

print(f'Retorno de la cartera: {retorno_cartera}')
print(f'Volatilidad de la cartera: {volatilidad_cartera}')
print(f'Ratio Sharpe de la cartera: {sharpe_cartera}')


retorno_cartera_modelo_1 = retorno_cartera
volatilidad_cartera_modelo_1 = volatilidad_cartera
sharpe_cartera_modelo_1 = sharpe_cartera

retorno_cartera_modelo_1

volatilidad_cartera_modelo_1

sharpe_cartera_modelo_1

pesos_cartera_optima_modelo_1 = {nombre: peso for nombre, peso in zip(valores.keys(), pesos_cartera_optima)}
pesos_cartera_optima_modelo_1

from scipy.optimize import minimize

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

# Función para calcular el retorno y la volatilidad
def calcular_retorno_volatilidad(datos):
    retorno = datos['Close'].pct_change().mean() * 252
    volatilidad = datos['Close'].pct_change().std() * np.sqrt(252)
    return retorno, volatilidad

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

# Calcular retorno y volatilidad
retorno_volatilidad = {nombre: calcular_retorno_volatilidad(datos) for nombre, datos in valores.items()}

for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    print(f'{nombre}: Retorno={retorno}, Volatilidad={volatilidad}')

# Crear matriz de covarianza
matriz_covarianza = np.zeros((len(valores), len(valores)))
for i, (_, datos_i) in enumerate(valores.items()):
    for j, (_, datos_j) in enumerate(valores.items()):
        matriz_covarianza[i][j] = datos_i['Close'].pct_change().cov(datos_j['Close'].pct_change()) * 252

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

# Encontrar la cartera con el ratio Sharpe máximo
indice_sharpe_max = np.argmax(resultados[:, len(valores) + 2])
pesos_sharpe_max = resultados[indice_sharpe_max, :len(valores)]
retorno_sharpe_max = resultados[indice_sharpe_max, len(valores)]
volatilidad_sharpe_max = resultados[indice_sharpe_max, len(valores) + 1]
sharpe_max = resultados[indice_sharpe_max, len(valores) + 2]

# Encontrar la cartera con el ratio Sharpe mínimo
indice_sharpe_min = np.argmin(resultados[:, len(valores) + 2])
pesos_sharpe_min = resultados[indice_sharpe_min, :len(valores)]
retorno_sharpe_min = resultados[indice_sharpe_min, len(valores)]
volatilidad_sharpe_min = resultados[indice_sharpe_min, len(valores) + 1]
sharpe_min = resultados[indice_sharpe_min, len(valores) + 2]

print(f'Ratio Sharpe Máximo: {sharpe_max}')
print(f'Retorno de la Cartera (Sharpe Máximo): {retorno_sharpe_max}')
print(f'Volatilidad de la Cartera (Sharpe Máximo): {volatilidad_sharpe_max}')
print('Pesos de la Cartera (Sharpe Máximo):')
for nombre, peso in zip(valores.keys(), pesos_sharpe_max):
    print(f'{nombre}: {peso:.2f}%')

print(f'Ratio Sharpe Mínimo: {sharpe_min}')
print(f'Retorno de la Cartera (Sharpe Mínimo): {retorno_sharpe_min}')
print(f'Volatilidad de la Cartera (Sharpe Mínimo): {volatilidad_sharpe_min}')
print('Pesos de la Cartera (Sharpe Mínimo):')
for nombre, peso in zip(valores.keys(), pesos_sharpe_min):
    print(f'{nombre}: {peso:.2f}%')

frontera_volatilidad = np.linspace(0, 0.3, 100)
frontera_retorno = []
for volatilidad in frontera_volatilidad:
    restricciones_frontera = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1},
                              {'type': 'eq', 'fun': lambda pesos: np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos))) - volatilidad})
    resultados_frontera = minimize(lambda pesos: -np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos), pesos_iniciales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones_frontera)
    retorno = np.dot([retorno for _, retorno in retorno_volatilidad.values()], resultados_frontera.x)
    frontera_retorno.append(retorno)

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

pesos_labels_max = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_sharpe_max * 100)]
fig_pie_max = go.Figure(data=[go.Pie(labels=pesos_labels_max, values=pesos_sharpe_max * 100, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
fig_pie_max.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
fig_pie_max.update_layout(title='Pesos de la Cartera Óptima (Sharpe Máximo)', showlegend=False)
fig_pie_max.show()

retorno_cartera_modelo_2 = retorno_sharpe_max
volatilidad_cartera_modelo_2 = volatilidad_sharpe_max
sharpe_cartera_modelo_2 = sharpe_max

retorno_cartera_modelo_2

volatilidad_cartera_modelo_2

sharpe_cartera_modelo_2

pesos_cartera_optima_modelo_2 = {nombre: peso for nombre, peso in zip(valores.keys(), pesos_sharpe_max)}
pesos_cartera_optima_modelo_2

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

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

# funcion calcular el retorno y la volatilidad
def calcular_retorno_volatilidad(datos):
    retorno = datos['Close'].pct_change().mean() * 252
    volatilidad = datos['Close'].pct_change().std() * np.sqrt(252)
    return retorno, volatilidad

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

# Calcular retorno y volatilidad
retorno_volatilidad = {nombre: calcular_retorno_volatilidad(datos) for nombre, datos in valores.items()}

for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    print(f'{nombre}: Retorno={retorno}, Volatilidad={volatilidad}')

matriz_covarianza = np.zeros((len(valores), len(valores)))
for i, (_, datos_i) in enumerate(valores.items()):
    for j, (_, datos_j) in enumerate(valores.items()):
        matriz_covarianza[i][j] = datos_i['Close'].pct_change().cov(datos_j['Close'].pct_change()) * 252

X = np.random.random((10000, len(valores)))
X /= np.sum(X, axis=1)[:, np.newaxis]

retornos = np.array([retorno for _, retorno in retorno_volatilidad.values()])
volatilidades = np.sqrt(np.dot(X, np.dot(matriz_covarianza, X.T)).diagonal())

ratios_sharpe = retornos @ X.T / volatilidades

gb = GradientBoostingRegressor(n_estimators=100, max_depth=3)
gb.fit(X, ratios_sharpe)

# Encontrar los pesos que maximizan el ratio Sharpe
pesos_optimales = X[np.argmax(gb.predict(X))]
retorno_optimo = retornos @ pesos_optimales
volatilidad_optima = np.sqrt(pesos_optimales.T @ np.dot(matriz_covarianza, pesos_optimales))
sharpe_max = retorno_optimo / volatilidad_optima

# Encontrar los pesos que minimizan el ratio Sharpe
pesos_minimos = X[np.argmin(gb.predict(X))]
retorno_minimo = retornos @ pesos_minimos
volatilidad_minima = np.sqrt(pesos_minimos.T @ np.dot(matriz_covarianza, pesos_minimos))
sharpe_min = retorno_minimo / volatilidad_minima

print(f'Ratio Sharpe Máximo: {sharpe_max}')
print(f'Retorno de la Cartera (Sharpe Máximo): {retorno_optimo}')
print(f'Volatilidad de la Cartera (Sharpe Máximo): {volatilidad_optima}')
print('Pesos de la Cartera (Sharpe Máximo):')
for nombre, peso in zip(valores.keys(), pesos_optimales):
    print(f'{nombre}: {peso:.2f}%')

print(f'Ratio Sharpe Mínimo: {sharpe_min}')
print(f'Retorno de la Cartera (Sharpe Mínimo): {retorno_minimo}')
print(f'Volatilidad de la Cartera (Sharpe Mínimo): {volatilidad_minima}')
print('Pesos de la Cartera (Sharpe Mínimo):')
for nombre, peso in zip(valores.keys(), pesos_minimos):
    print(f'{nombre}: {peso:.2f}%')

frontera_volatilidad = np.linspace(0, 0.3, 100)
frontera_retorno = []
for volatilidad in frontera_volatilidad:
    restricciones_frontera = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1},
                              {'type': 'eq', 'fun': lambda pesos: np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos))) - volatilidad})
    resultados_frontera = minimize(lambda pesos: -np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos), pesos_optimales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones_frontera)
    retorno = np.dot([retorno for _, retorno in retorno_volatilidad.values()], resultados_frontera.x)
    frontera_retorno.append(retorno)

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

pesos_labels_max = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_optimales * 100)]
fig_pie_max = go.Figure(data=[go.Pie(labels=pesos_labels_max, values=pesos_optimales * 100, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
fig_pie_max.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
fig_pie_max.update_layout(title='Pesos de la Cartera Óptima (Sharpe Máximo)', showlegend=False)
fig_pie_max.show()

retorno_cartera_modelo_3 = retorno_optimo
volatilidad_cartera_modelo_3 = volatilidad_optima
sharpe_cartera_modelo_3 = sharpe_max

retorno_cartera_modelo_3

volatilidad_cartera_modelo_3

sharpe_cartera_modelo_3

pesos_cartera_optima_modelo_3 = {nombre: peso for nombre, peso in zip(valores.keys(), pesos_optimales)}
pesos_cartera_optima_modelo_3

import xgboost as xgb

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

# Función calcular el retorno y la volatilidad
def calcular_retorno_volatilidad(datos):
    retorno = datos['Close'].pct_change().mean() * 252
    volatilidad = datos['Close'].pct_change().std() * np.sqrt(252)
    return retorno, volatilidad

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

retorno_volatilidad = {nombre: calcular_retorno_volatilidad(datos) for nombre, datos in valores.items()}

for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    print(f'{nombre}: Retorno={retorno}, Volatilidad={volatilidad}')

matriz_covarianza = np.zeros((len(valores), len(valores)))
for i, (_, datos_i) in enumerate(valores.items()):
    for j, (_, datos_j) in enumerate(valores.items()):
        matriz_covarianza[i][j] = datos_i['Close'].pct_change().cov(datos_j['Close'].pct_change()) * 252

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

X = resultados[:, :len(valores)]
y = resultados[:, len(valores) + 2]

dtrain = xgb.DMatrix(X, label=y)
params = {'max_depth': 3, 'eta': 0.1, 'objective': 'reg:squarederror'}
num_round = 100
bst = xgb.train(params, dtrain, num_round)

dtest = xgb.DMatrix(X)
predicciones = bst.predict(dtest)

pesos_optimales = X[np.argmax(predicciones)]
retorno_optimo = np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos_optimales)
volatilidad_optima = np.sqrt(np.dot(pesos_optimales.T, np.dot(matriz_covarianza, pesos_optimales)))
sharpe_max = retorno_optimo / volatilidad_optima

pesos_minimos = X[np.argmin(predicciones)]
retorno_minimo = np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos_minimos)
volatilidad_minima = np.sqrt(np.dot(pesos_minimos.T, np.dot(matriz_covarianza, pesos_minimos)))
sharpe_min = retorno_minimo / volatilidad_minima

print(f'Ratio Sharpe Máximo: {sharpe_max}')
print(f'Retorno de la Cartera (Sharpe Máximo): {retorno_optimo}')
print(f'Volatilidad de la Cartera (Sharpe Máximo): {volatilidad_optima}')
print('Pesos de la Cartera (Sharpe Máximo):')
for nombre, peso in zip(valores.keys(), pesos_optimales):
    print(f'{nombre}: {peso:.2f}%')

print(f'Ratio Sharpe Mínimo: {sharpe_min}')
print(f'Retorno de la Cartera (Sharpe Mínimo): {retorno_minimo}')
print(f'Volatilidad de la Cartera (Sharpe Mínimo): {volatilidad_minima}')
print('Pesos de la Cartera (Sharpe Mínimo):')
for nombre, peso in zip(valores.keys(), pesos_minimos):
    print(f'{nombre}: {peso:.2f}%')

frontera_volatilidad = np.linspace(0, 0.3, 100)
frontera_retorno = []
for volatilidad in frontera_volatilidad:
    restricciones_frontera = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1},
                              {'type': 'eq', 'fun': lambda pesos: np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos))) - volatilidad})
    resultados_frontera = minimize(lambda pesos: -np.dot([retorno for _, retorno in retorno_volatilidad.values()], pesos), pesos_optimales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones_frontera)
    retorno = np.dot([retorno for _, retorno in retorno_volatilidad.values()], resultados_frontera.x)
    frontera_retorno.append(retorno)

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

pesos_labels_max = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_optimales * 100)]
fig_pie_max = go.Figure(data=[go.Pie(labels=pesos_labels_max, values=pesos_optimales * 100, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
fig_pie_max.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
fig_pie_max.update_layout(title='Pesos de la Cartera Óptima (Sharpe Máximo)', showlegend=False)
fig_pie_max.show()

retorno_cartera_modelo_4 = retorno_optimo
volatilidad_cartera_modelo_4 = volatilidad_optima
sharpe_cartera_modelo_4 = sharpe_max

retorno_cartera_modelo_4

volatilidad_cartera_modelo_4

sharpe_cartera_modelo_4

pesos_cartera_optima_modelo_4 = {nombre: peso for nombre, peso in zip(valores.keys(), pesos_optimales)}
pesos_cartera_optima_modelo_4

from sklearn.decomposition import PCA

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

# Función para calcular el retorno y la volatilidad
def calcular_retorno_volatilidad(datos):
    retorno = datos['Close'].pct_change().mean() * 252
    volatilidad = datos['Close'].pct_change().std() * np.sqrt(252)
    return retorno, volatilidad

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

retorno_volatilidad = {nombre: calcular_retorno_volatilidad(datos) for nombre, datos in valores.items()}

for nombre, (retorno, volatilidad) in retorno_volatilidad.items():
    print(f'{nombre}: Retorno={retorno}, Volatilidad={volatilidad}')

matriz_covarianza = np.zeros((len(valores), len(valores)))
for i, (_, datos_i) in enumerate(valores.items()):
    for j, (_, datos_j) in enumerate(valores.items()):
        matriz_covarianza[i][j] = datos_i['Close'].pct_change().cov(datos_j['Close'].pct_change()) * 252

# Alinear los índices de los DataFrames para asegurar que todas las series temporales tengan la misma longitud
retornos_diarios_df = pd.concat([datos['Close'].pct_change().dropna() for _, datos in valores.items()], axis=1, join='inner')
retornos_diarios_df.columns = valores.keys()

# Convertir DataFrame a matriz numpy
retornos_diarios = retornos_diarios_df.values

# Aplicar PCA
pca = PCA()
pca.fit(retornos_diarios)
componentes_principales = pca.transform(retornos_diarios)

# Reconstruir los datos utilizando solo las componentes principales más significativas
num_componentes = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.95) + 1
retornos_reconstruidos = np.dot(componentes_principales[:, :num_componentes], pca.components_[:num_componentes, :])
retornos_reconstruidos += pca.mean_

# Calcular el retorno y la volatilidad de la cartera utilizando las componentes principales
retornos_medios = retornos_reconstruidos.mean(axis=0) * 252
volatilidades = retornos_reconstruidos.std(axis=0) * np.sqrt(252)
matriz_covarianza_pca = np.cov(retornos_reconstruidos.T) * 252

# Función objetivo para minimizar la volatilidad de la cartera
def objetivo(pesos):
    return np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza_pca, pesos)))

# Restricciones para los pesos de la cartera
restricciones = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1})

# Optimización
pesos_iniciales = np.ones(len(valores)) / len(valores)
resultados = minimize(objetivo, pesos_iniciales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones)

# Calcular retorno, volatilidad y ratio Sharpe de la cartera óptima
retorno_cartera = np.dot(retornos_medios, resultados.x)
volatilidad_cartera = np.sqrt(np.dot(resultados.x.T, np.dot(matriz_covarianza_pca, resultados.x)))
sharpe_cartera = retorno_cartera / volatilidad_cartera

# Encontrar los pesos que maximizan el ratio Sharpe
pesos_sharpe_max = resultados.x
retorno_sharpe_max = retorno_cartera
volatilidad_sharpe_max = volatilidad_cartera
sharpe_max = sharpe_cartera

# Encontrar los pesos que minimizan el ratio Sharpe (maximizar la penalización negativa)
resultados_min = minimize(lambda pesos: -np.dot(retornos_medios, pesos) / np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza_pca, pesos))), pesos_iniciales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones)
pesos_sharpe_min = resultados_min.x
retorno_sharpe_min = np.dot(retornos_medios, pesos_sharpe_min)
volatilidad_sharpe_min = np.sqrt(np.dot(pesos_sharpe_min.T, np.dot(matriz_covarianza_pca, pesos_sharpe_min)))
sharpe_min = retorno_sharpe_min / volatilidad_sharpe_min

print(f'Ratio Sharpe Máximo: {sharpe_max}')
print(f'Retorno de la Cartera (Sharpe Máximo): {retorno_sharpe_max}')
print(f'Volatilidad de la Cartera (Sharpe Máximo): {volatilidad_sharpe_max}')
print('Pesos de la Cartera (Sharpe Máximo):')
for nombre, peso in zip(valores.keys(), pesos_sharpe_max):
    print(f'{nombre}: {peso:.2f}%')

print(f'Ratio Sharpe Mínimo: {sharpe_min}')
print(f'Retorno de la Cartera (Sharpe Mínimo): {retorno_sharpe_min}')
print(f'Volatilidad de la Cartera (Sharpe Mínimo): {volatilidad_sharpe_min}')
print('Pesos de la Cartera (Sharpe Mínimo):')
for nombre, peso in zip(valores.keys(), pesos_sharpe_min):
    print(f'{nombre}: {peso:.2f}%')

frontera_volatilidad = np.linspace(0, 0.3, 100)
frontera_retorno = []
for volatilidad in frontera_volatilidad:
    restricciones_frontera = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1},
                              {'type': 'eq', 'fun': lambda pesos: np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza_pca, pesos))) - volatilidad})
    resultados_frontera = minimize(lambda pesos: -np.dot(retornos_medios, pesos), pesos_iniciales, method='SLSQP', bounds=[(0, 1)] * len(valores), constraints=restricciones_frontera)
    retorno = np.dot(retornos_medios, resultados_frontera.x)
    frontera_retorno.append(retorno)

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

pesos_labels_max = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_sharpe_max * 100)]
fig_pie_max = go.Figure(data=[go.Pie(labels=pesos_labels_max, values=pesos_sharpe_max * 100, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
fig_pie_max.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
fig_pie_max.update_layout(title='Pesos de la Cartera Óptima (Sharpe Máximo)', showlegend=False)
fig_pie_max.show()

retorno_cartera_modelo_5 = retorno_sharpe_max
volatilidad_cartera_modelo_5 = volatilidad_sharpe_max
sharpe_cartera_modelo_5 = sharpe_max

retorno_cartera_modelo_5

volatilidad_cartera_modelo_5

sharpe_cartera_modelo_5

pesos_cartera_optima_modelo_5 = {nombre: peso for nombre, peso in zip(valores.keys(), pesos_sharpe_max)}
pesos_cartera_optima_modelo_5

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

df_comparacion = pd.DataFrame({
    'Modelo': resultados_modelos.keys(),
    'Retorno': [resultados_modelos[modelo]['retorno'] for modelo in resultados_modelos],
    'Volatilidad': [resultados_modelos[modelo]['volatilidad'] for modelo in resultados_modelos],
    'Ratio Sharpe': [resultados_modelos[modelo]['sharpe'] for modelo in resultados_modelos]
})

df_comparacion.set_index('Modelo', inplace=True)

print(df_comparacion)

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

pesos_modelos = {
    'Modelo 1': pesos_cartera_optima_modelo_1,
    'Modelo 2': {k: v * 100 for k, v in pesos_cartera_optima_modelo_2.items()},
    'Modelo 3': {k: v * 100 for k, v in pesos_cartera_optima_modelo_3.items()},
    'Modelo 4': {k: v * 100 for k, v in pesos_cartera_optima_modelo_4.items()},
    'Modelo 5': {k: v * 100 for k, v in pesos_cartera_optima_modelo_5.items()}
}

# DataFrame para la comparación de pesos
df_pesos = pd.DataFrame(pesos_modelos)
df_pesos.index.name = 'Activos'
df_pesos.reset_index(inplace=True)
print(df_pesos)

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

if __name__ == "__main__":
    main()
