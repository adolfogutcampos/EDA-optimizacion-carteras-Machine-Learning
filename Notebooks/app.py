
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px

st.title('EDA Optimización de Carteras con Machine Learning')

st.markdown('''# Indice:''')
st.markdown('''# 0.- Introduccion''')
st.markdown('''# 1.- Importar librerias necesarias''')
st.markdown('''## 1.1.- Importar datasets de archivos del repositorio''')
st.markdown('''# 2.- Principios del EDA (Exploratory Data Analysis)''')
st.markdown('''## 2.1.- Contexto''')
st.markdown('''## 2.2.- Hipotesis''')
st.markdown('''# 3.- Cotizacion y rentabilidad de conjunto de 35 empresas y 10 indices''')
st.markdown('''## 3.1.- Cotizaciones''')
st.markdown('''### 3.1.1.- Cotizaciones de las 35 empresas''')
st.markdown('''### 3.1.2.- Cotizaciones de los 10 indices bursatiles''')
st.markdown('''## 3.2.- Rentabilidades''')
st.markdown('''### 3.2.1.- Rentabilidad de 35 empresas''')
st.markdown('''### 3.2.2.- Rentabilidad de 10 indices''')
st.markdown('''# 4.- Seleccion de muestra de empresas e indices para una cartera de 10 valores''')
st.markdown('''# 5.- Cartera optimizada con Sharpe''')
st.markdown('''## 5.1.- Teoria''')
st.markdown('''## 5.2.- Portfolio con cartera optimizada sin modelos de machine learning (Modelo 1)''')
st.markdown('''### Resultados finales modelo 1:''')
st.markdown('''## 5.3.- Portfolio con cartera optimizada con simulacion de Monte Carlo (Modelo 2)''')
st.markdown('''### Resultados finales modelo 2:''')
st.markdown('''# 6.- Carteras optimizadas con Modelos de Machine Learning''')
st.markdown('''## 6.1.- Modelo Supervisado ML con Gradient Boosting (Modelo 3)''')
st.markdown('''### Resultados finales modelo 3:''')
st.markdown('''## 6.2.- Modelo Supervisado ML con XGBoost (Modelo 4)''')
st.markdown('''### Resultados finales modelo 4:''')
st.markdown('''## 6.3.- Modelo No Supervisado PCA (Principal Component Analysis) (Modelo 5)''')
st.markdown('''### Resultados finales modelo 5:''')
st.markdown('''# 7.- Comparacion de resultados finales de los 5 modelos''')
st.markdown('''## 7.1.- Comparacion de retorno, volatilidad y sharpe de las 5 carteras''')
st.markdown('''## 7.2.- Comparacion de peso de carteras''')

st.pyplot(fig=plt.figure())

def plot_code():
    fig = px.line(
        all_data,
        x='Date',
        y='Close',
        color='Company',
        title='Precio de cierre (Close) en el período 01/01/2018 - 20/05/2024',
        labels={'Close': 'Precio de cierre (EUR)', 'Date': 'Fecha'}
    )
    
    fig.show()

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
    fig = px.line(
        all_index_data,
        x='Date',
        y='Close',
        color='Index',
        title='Valor de cierre de los 10 índices entre 01/01/2018 y 20/05/2024',
        labels={'Close': 'Valor de cierre (puntos)', 'Date': 'Fechas'}
    )
    
    fig.show()

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
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

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
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

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
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

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
    pesos_cartera_optima_pct = resultados.x * 100
    pesos_labels = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_cartera_optima_pct)]
    fig_pie = go.Figure(data=[go.Pie(labels=pesos_labels, values=pesos_cartera_optima_pct, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
    fig_pie.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
    fig_pie.update_layout(title='Pesos de la cartera óptima', showlegend=False)
    fig_pie.show()

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
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

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
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

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
    pesos_labels_max = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_sharpe_max * 100)]
    fig_pie_max = go.Figure(data=[go.Pie(labels=pesos_labels_max, values=pesos_sharpe_max * 100, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
    fig_pie_max.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
    fig_pie_max.update_layout(title='Pesos de la Cartera Óptima (Sharpe Máximo)', showlegend=False)
    fig_pie_max.show()

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
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

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
    pesos_labels_max = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_optimales * 100)]
    fig_pie_max = go.Figure(data=[go.Pie(labels=pesos_labels_max, values=pesos_optimales * 100, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
    fig_pie_max.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
    fig_pie_max.update_layout(title='Pesos de la Cartera Óptima (Sharpe Máximo)', showlegend=False)
    fig_pie_max.show()

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
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

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
    pesos_labels_max = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_optimales * 100)]
    fig_pie_max = go.Figure(data=[go.Pie(labels=pesos_labels_max, values=pesos_optimales * 100, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
    fig_pie_max.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
    fig_pie_max.update_layout(title='Pesos de la Cartera Óptima (Sharpe Máximo)', showlegend=False)
    fig_pie_max.show()

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
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

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
    pesos_labels_max = [f'{nombre}: {peso:.2f}%' for nombre, peso in zip(valores.keys(), pesos_sharpe_max * 100)]
    fig_pie_max = go.Figure(data=[go.Pie(labels=pesos_labels_max, values=pesos_sharpe_max * 100, textinfo='label+percent', texttemplate='%{label}', marker=dict(colors=px.colors.qualitative.Bold))])
    fig_pie_max.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
    fig_pie_max.update_layout(title='Pesos de la Cartera Óptima (Sharpe Máximo)', showlegend=False)
    fig_pie_max.show()

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
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

plot_code()

st.pyplot(fig=plt.figure())

def plot_code():
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

plot_code()
