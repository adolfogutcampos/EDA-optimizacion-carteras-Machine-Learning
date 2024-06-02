
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Análisis Exploratorio de Datos - Optimización de Carteras")

# Cargar los datos
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Asegúrate de que las rutas sean correctas
ANA_dataset = load_data('Notebooks/data/ANA_dataset.csv')
ANE_dataset = load_data('Notebooks/data/ANE_dataset.csv')

# Mostrar los datos
st.write(ANA_dataset.head())
st.write(ANE_dataset.head())

# Gráficos de ejemplo
def plot_data(df, column_name):
    st.write(f"Gráfico de {column_name}")
    fig, ax = plt.subplots()
    ax.hist(df[column_name])
    st.pyplot(fig)

# Ajusta las columnas según tus datos
plot_data(ANA_dataset, 'nombre_de_tu_columna')
plot_data(ANE_dataset, 'nombre_de_tu_columna')
