
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings

warnings.filterwarnings('ignore')

# Cargar los datos
ANA_dataset = pd.read_csv('data/ANA_dataset.csv')
ANE_dataset = pd.read_csv('data/ANE_dataset.csv')
ACS_dataset = pd.read_csv('data/ACS_dataset.csv')
IBE_dataset = pd.read_csv('data/IBE_dataset.csv')
TEF_dataset = pd.read_csv('data/TEF_dataset.csv')

# Encabezados
st.title("Investment Portfolio Analysis")
st.write("""
# Análisis de 5 modelos de portafolios de inversión
""")

# Mostrar los primeros registros de cada dataset
st.write("## Acciona Dataset")
st.write(ANA_dataset.head())

st.write("## Acciona Energías Dataset")
st.write(ANE_dataset.head())

st.write("## ACS Dataset")
st.write(ACS_dataset.head())

st.write("## Iberdrola Dataset")
st.write(IBE_dataset.head())

st.write("## Telefónica Dataset")
st.write(TEF_dataset.head())

# Ejemplo de gráfico
st.write("## Example Plot")
fig, ax = plt.subplots()
ax.plot(ANA_dataset['column_name'])  # Reemplaza 'column_name' con el nombre real de la columna
st.pyplot(fig)

# Ejemplo de gráfico interactivo con Plotly
st.write("## Interactive Plotly Chart")
fig = go.Figure(data=go.Scatter(x=ANA_dataset['column_x'], y=ANA_dataset['column_y']))  # Reemplaza 'column_x' y 'column_y' con los nombres reales de las columnas
st.plotly_chart(fig)

# Añade aquí más análisis y gráficos según sea necesario
