
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px

st.title("Análisis de Optimización de Carteras")

st.markdown("# 2.1. Introducción")
st.markdown("En este proyecto analizamos diferentes métodos de optimización de carteras utilizando modelos de Machine Learning y técnicas tradicionales.")

st.markdown("## 2.1.1. Fundamentos de Optimización de Carteras")
st.markdown("La optimización de carteras es el proceso de elegir la distribución de activos que maximiza el retorno esperado para un nivel dado de riesgo. En este análisis, comparamos cinco modelos diferentes para optimizar una cartera de 10 activos.")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("## 2.1.2. Modelos Utilizados")
st.markdown("Se utilizan cinco modelos para la optimización de carteras: Modelo 1 - Igual Ponderación, Modelo 2 - Frontera Eficiente de Markowitz, Modelo 3 - Gradient Boosting, Modelo 4 - Regresión LASSO, Modelo 5 - Random Forest.")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("## 2.1.3. Resultados y Comparación")
st.markdown("Finalmente, comparamos los resultados de los modelos en términos de retorno, volatilidad y ratio Sharpe.")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("### Paso 1: Calcular Retorno y Volatilidad para Cada Valor")
st.markdown("Calculamos el retorno anualizado y la volatilidad anualizada para cada uno de los 10 valores. Utilizamos la media y la desviación estándar de los cambios porcentuales diarios (retornos diarios) multiplicados por 252 (el número aproximado de días de trading en un año).")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("### Paso 2: Crear la Matriz de Covarianza")
st.markdown("La matriz de covarianza mide cómo los retornos de los diferentes valores se mueven juntos. Esta matriz es esencial para calcular la volatilidad de la cartera.")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("### Paso 3: Generar Simulaciones de Portafolios")
st.markdown("Generamos 10,000 combinaciones de pesos al azar para los valores en la cartera y calculamos el retorno esperado, la volatilidad y el ratio Sharpe para cada combinación. Luego, graficamos los resultados.")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("### Paso 4: Frontera Eficiente de Markowitz")
st.markdown("Calculamos la frontera eficiente de Markowitz, que representa las carteras que ofrecen el máximo retorno para un nivel dado de riesgo. Graficamos los 10,000 portafolios generados, los puntos de los valores individuales y la frontera eficiente de Markowitz.")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("### Paso 5: Crear Gráfico Interactivo Circular con los Pesos de la Cartera Óptima")
st.markdown("Finalmente, creamos un gráfico circular para visualizar los pesos de la cartera óptima (con el ratio Sharpe máximo).")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("### Resultados finales modelo 2:")
st.markdown("# 6.- Carteras optimizadas con Modelos de Machine Learning")
st.markdown("## 6.1.- Modelo Supervisado ML con Gradient Boosting (Modelo 3)")
st.markdown("Paso 1: Calcular Retorno y Volatilidad para Cada Valor")
st.markdown("Calculamos el retorno anualizado y la volatilidad anualizada para cada uno de los 10 valores. Utilizamos la media y la desviación estándar de los cambios porcentuales diarios (retornos diarios) multiplicados por 252 (el número aproximado de días de trading en un año).")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("Paso 2: Crear la Matriz de Covarianza")
st.markdown("La matriz de covarianza mide cómo los retornos de los diferentes valores se mueven juntos. Esta matriz es esencial para calcular la volatilidad de la cartera.")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("Paso 3: Optimización de la Cartera usando Gradient Boosting")
st.markdown("Generamos 10,000 combinaciones de pesos al azar para los valores en la cartera y calculamos el retorno esperado, la volatilidad y el ratio Sharpe para cada combinación. Luego, entrenamos un modelo de Gradient Boosting para predecir el ratio Sharpe en función de los pesos y utilizamos el modelo para encontrar los pesos que maximizan y minimizan el ratio Sharpe.")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("Paso 4: Mostrar Resultados")
st.markdown("Mostramos los resultados de la optimización, incluyendo los pesos de los valores en la cartera, el retorno, la volatilidad y el ratio Sharpe.")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("Paso 5: Frontera Eficiente de Markowitz")
st.markdown("Calculamos la frontera eficiente de Markowitz, que representa las carteras que ofrecen el máximo retorno para un nivel dado de riesgo. Utilizamos optimi")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("Paso 6: Graficar la Frontera Eficiente y los Resultados")
st.markdown("Graficamos los 10,000 portafolios generados, los puntos de los valores individuales y la frontera eficiente de Markowitz.")

# Insertar aquí el código de los gráficos correspondientes

st.markdown("Paso 7: Crear Gráfico Interactivo Circular con los Pesos de la Cartera Óptima")
st.markdown("Finalmente, creamos un gráfico circular para visualizar los pesos de la cartera óptima (con el ratio Sharpe máximo).")

# Insertar aquí el código de los gráficos correspondientes
