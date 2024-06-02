
import streamlit as st
import matplotlib.pyplot as plt

# Títulos y gráficos de ejemplo

# Markdown Title
st.markdown('''# Indice:

[0.- Introduccion](#0--introduccion)

[1.- Importar librerias necesarias](#1--importar-librerias-necesarias)

   - [1.1.- Importar datasets de archivos del repositorio](#11--importar-datasets-de-archivos-del-repositorio)

[2.- Principios del EDA (Exploratory Data Analysis)](#2--principios-del-eda-exploratory-data-analysis)

   - [2.1.- Contexto](#21--contexto)

   - [2.2.- Hipotesis](#22--hipotesis)

[3.- Cotizacion y rentabilidad de conjunto de 35 empresas y 10 indices](#3--cotizacion-y-rentabilidad-de-conjunto-de-35-empresas-y-10-indices)

   - [3.1.- Cotizaciones](#31--cotizaciones)

     - [3.1.1.- Cotizaciones de las 35 empresas](#311--cotizaciones-de-las-35-empresas)

     - [3.1.2.- Cotizaciones de los 10 indices bursatiles](#312--cotizaciones-de-los-10-indices-bursatiles)

   - [3.2.- Rentabilidades](#32--rentabilidades)

     - [3.2.1.- Rentabilidad de 35 empresas](#321--rentabilidad-de-35-empresas)

     - [3.2.2.- Rentabilidad de 10 indices](#322--rentabilidad-de-10-indices)

[4.- Seleccion de muestra de empresas e indices para una cartera de 10 valores](#4--seleccion-de-muestra-de-empresas-e-indices-para-una-cartera-de-10-valores)

[5.- Cartera optimizada con Sharpe](#5--cartera-optimizada-con-sharpe)

   - [5.1.- Teoria](#51--teoria)

   - [5.2.- Portfolio con cartera optimizada sin modelos de machine learning (Modelo 1)](#52--portfolio-con-cartera-optimizada-sin-modelos-de-machine-learning-modelo-1)

   - [5.3.- Portfolio con cartera optimizada con simulacion de Monte Carlo (Modelo 2)](#53--portfolio-con-cartera-optimizada-con-simulacion-de-monte-carlo-modelo-2)

[6.- Carteras optimizadas con Modelos de Machine Learning](#6--carteras-optimizadas-con-modelos-de-machine-learning)

   - [6.1.- Modelo Supervisado ML con Gradient Boosting (Modelo 3)](#61--modelo-supervisado-ml-con-gradient-boosting-modelo-3)

   - [6.2.- Modelo Supervisado ML con XGBoost (Modelo 4)](#62--modelo-supervisado-ml-con-xgboost-modelo-4)

   - [6.3.- Modelo No Supervisado PCA (Principal Component Analysis) (Modelo 5)](#63--modelo-no-supervisado-pca-principal-component-analysis-modelo-5)

[7.- Comparacion de resultados finales de los 5 modelos](#7--comparacion-de-resultados-finales-de-los-5-modelos)

   - [7.1.- Comparacion de retorno, volatilidad y sharpe de las 5 carteras](#71--comparacion-de-retorno-volatilidad-y-sharpe-de-las-5-carteras)
   
   - [7.2.- Comparacion de peso de carteras](#72--comparacion-de-peso-de-carteras)

''')

# Markdown Title
st.markdown('''# 0.- Introduccion

El análisis de datos financieros es una herramienta crucial en el mundo de las inversiones. Proporciona información valiosa sobre el comportamiento del mercado y ayuda a los inversores a tomar decisiones informadas. 

Este Análisis Exploratorio de Datos (EDA) se centra en las 35 empresas que conforman el IBEX 35 y 10 índices bursátiles globales. El objetivo principal es optimizar carteras de inversión mediante diversas técnicas y modelos, evaluando su rendimiento en términos de retorno, volatilidad y ratio Sharpe.

De entre las 35 empresas y 10 indices, se tomará 5 empresas y 5 indices para desarrollar carteras optimizadas.''')

# Markdown Title
st.markdown('''# 1.- Importar librerias necesarias''')

# Markdown Title
st.markdown('''## 1.1.- Importar datasets de archivos del repositorio''')

# Markdown Title
st.markdown('''Aquí vamos a realizar la lectura (pd.read_csv) de los datasets de las 35 empresas.''')

# Markdown Title
st.markdown('''De entre el listado de las 35 empresas, vamos a ver las columnas de ACS como ejemplo:''')

# Markdown Title
st.markdown('''De los 10 indices bursátiles, vamos a ver las columnas del indice del NIFTY50 como ejemplo:''')

# Markdown Title
st.markdown('''# 2.- Principios del EDA (Exploratory Data Analysis)''')

# Markdown Title
st.markdown('''## 2.1.- Contexto''')

# Markdown Title
st.markdown('''El IBEX 35 es el principal índice bursátil de la Bolsa de Madrid, que agrupa a las 35 empresas más líquidas y de mayor capitalización de mercado de España. Además, los índices bursátiles seleccionados para este análisis, como el S&P 500, NASDAQ, y NIFTY 50, representan los principales mercados financieros a nivel mundial. 

El análisis de estos datos permite una visión amplia del comportamiento del mercado tanto a nivel local como global.''')

# Markdown Title
st.markdown('''## 2.2.- Hipotesis

1- **Diversificación**: ¿Puede una cartera diversificada reducir la volatilidad general y mejorar el ratio Sharpe?

2- **Eficiencia de Mercado**: ¿Cómo impacta la inclusión de valores de distintos mercados globales en la estabilidad y el rendimiento de la cartera?

3- **Modelos Supervisados**: ¿Pueden los modelos de machine learning supervisados (Gradient Boosting y XGBoost) identificar patrones complejos en los datos históricos de precios y mejorar la optimización de la cartera?

4- **Simulación de Monte Carlo**: ¿Generar múltiples combinaciones de pesos aleatorios puede proporcionar configuraciones óptimas sin necesidad de un modelo supervisado?

5- **Reducción de Dimensionalidad con PCA**: ¿La utilización de PCA para reducir la dimensionalidad del problema puede simplificar la optimización de la cartera sin perder información crucial?''')

# Markdown Title
st.markdown('''# 3.- Cotizacion y rentabilidad de conjunto de 35 empresas y 10 indices''')

# Markdown Title
st.markdown('''## 3.1.- Cotizaciones''')

# Markdown Title
st.markdown('''En el apartado de cotizaciones, se va a proceder a separar los gráficos de cotizaciones de las 35 empresas y el de los 10 índices bursátiles. Esto es debido a que las empresas del IBEX 35 cotizan en euros y tienen a tener una valoración alrededor de máximo 100 euros. Mientras que la cotización de los índices bursátiles se muestra en puntos, y su cotización puede ser de miles de puntos.

En otras palabras, se realiza una separación de gráficos para diferenciar un gráfico por divisa en EUR, y diferenciar otro gráfico basado en puntos.''')

# Markdown Title
st.markdown('''### 3.1.1.- Cotizaciones de las 35 empresas''')

# Markdown Title
st.markdown('''### 3.1.2.- Cotizaciones de los 10 indices bursatiles''')

# Markdown Title
st.markdown('''## 3.2.- Rentabilidades''')

# Markdown Title
st.markdown('''### 3.2.1.- Rentabilidad de 35 empresas''')

# Markdown Title
st.markdown('''### 3.2.2.- Rentabilidad de 10 indices''')

# Markdown Title
st.markdown('''# 4.- Seleccion de muestra de empresas e indices para una cartera de 10 valores''')

# Markdown Title
st.markdown('''Es aquí cuando vamos a escoger una empresa con el fin de analizarla de forma individual.

Para ello, vamos a crear una variable llamada dataset_empresa, la cual va a contener el dataset de la empresa que elijamos de entre las 35.

Como ejemplo, vamos a escoger el dataset de ACS. Generando una variable paralela mediante:

dataset_empresa = ACS_dataset

No obstante, podríamos cambiar el dataset al de otra empresa para realizar el analisis, cambiando en la siguiente celda las siglas de ACS por las de otra empresa. Como por ejemplo: ITX, quedando así como resultado: 

dataset_empresa = ITX_dataset''')

# Markdown Title
st.markdown('''En definitiva, la cartera que se va a generar está compuesta por los siguientes valores:''')

# Markdown Title
st.markdown('''# 5.- Cartera optimizada con Sharpe''')

# Markdown Title
st.markdown('''## 5.1.- Teoria''')

# Markdown Title
st.markdown('''## 5.2.- Portfolio con cartera optimizada sin modelos de machine learning (Modelo 1)''')

# Markdown Title
st.markdown('''Paso 1: Calcular retorno y volatilidad para cada valor

Calculamos el retorno anualizado y la volatilidad anualizada para cada valor. Utilizamos la media y la desviación estándar de los cambios porcentuales diarios (retornos diarios) multiplicados por 252 (el número aproximado de días de trading en un año).''')

# Markdown Title
st.markdown('''Paso 2: Mostrar retorno y volatilidad de cada valor

Mostramos los retornos y las volatilidades calculadas para cada valor.''')

# Markdown Title
st.markdown('''Paso 3: Crear matriz de covarianza

La matriz de covarianza mide cómo los retornos de los diferentes valores se mueven juntos. Esta matriz es crucial para calcular la volatilidad de la cartera.''')

# Markdown Title
st.markdown('''Paso 4: Función objetivo para minimizar la volatilidad de la cartera

Definimos una función objetivo que calcula la volatilidad de la cartera en función de los pesos asignados a cada valor. Esta función se utilizará en la optimización.''')

# Markdown Title
st.markdown('''Paso 5: Restricciones para los pesos de la cartera

Definimos una restricción que asegura que la suma de los pesos sea igual a 1, es decir, que se invierta el 100% del capital.''')

# Markdown Title
st.markdown('''Paso 6: Optimizar la cartera para maximizar el ratio Sharpe

Utilizamos el método de optimización SLSQP para encontrar los pesos que minimizan la volatilidad, respetando la restricción definida.''')

# Markdown Title
st.markdown('''Paso 7: Calcular retorno, volatilidad y ratio Sharpe de la cartera óptima

Calculamos el retorno esperado, la volatilidad y el ratio Sharpe de la cartera utilizando los pesos óptimos encontrados.''')

# Markdown Title
st.markdown('''Paso 8: Mostrar resultado

Mostramos el retorno, la volatilidad y el ratio Sharpe de la cartera óptima.''')

# Markdown Title
st.markdown('''Paso 9: Mostrar pesos de la cartera óptima

Mostramos los pesos asignados a cada valor en la cartera óptima.''')

# Markdown Title
st.markdown('''Paso 10: Calcular y mostrar la frontera eficiente de Markowitz

Calculamos la frontera eficiente, que representa las carteras que ofrecen el máximo retorno para un nivel dado de riesgo.''')

# Markdown Title
st.markdown('''Paso 11: Graficar la frontera eficiente de Markowitz

Creamos un gráfico para mostrar la frontera eficiente y los puntos de los valores individuales''')

# Markdown Title
st.markdown('''Paso 12: Crear gráfico interactivo circular con los pesos de la cartera óptima

Creamos un gráfico circular para visualizar los pesos de la cartera óptima.''')

# Markdown Title
st.markdown('''### Resultados finales modelo 1:''')

# Markdown Title
st.markdown('''## 5.3.- Portfolio con cartera optimizada con simulacion de Monte Carlo (Modelo 2)''')

# Markdown Title
st.markdown('''Paso 1: Calcular Retorno y Volatilidad para Cada Valor

En este paso, calculamos el retorno anualizado y la volatilidad anualizada para cada uno de los valores. Utilizamos la media y la desviación estándar de los cambios porcentuales diarios (retornos diarios) multiplicados por 252 (el número aproximado de días de trading en un año).''')

# Markdown Title
st.markdown('''Paso 2: Crear la Matriz de Covarianza

La matriz de covarianza mide cómo se mueven juntos los retornos de los diferentes valores. Es crucial para calcular la volatilidad de la cartera.''')

# Markdown Title
st.markdown('''Paso 3: Simulación de Montecarlo (Podemos cambiar el número de 10000 simulaciones)

En este paso, generamos 10,000 carteras aleatorias (podemos cambiar este número). Para cada cartera, calculamos el retorno esperado, la volatilidad esperada y el ratio Sharpe.''')

# Markdown Title
st.markdown('''Paso 4: Encontrar la Cartera con el Ratio Sharpe Máximo y Mínimo

Aquí identificamos las carteras con el ratio Sharpe máximo y mínimo entre las simuladas. El ratio Sharpe mide el rendimiento ajustado por el riesgo.''')

# Markdown Title
st.markdown('''Paso 5: Mostrar Resultados

Se presentan los resultados, incluyendo los pesos de cada valor en las carteras óptimas (tanto la de Sharpe máximo como la de Sharpe mínimo), así como los respectivos retornos y volatilidades.''')

# Markdown Title
st.markdown('''Paso 6: Frontera Eficiente de Markowitz

Calculamos la frontera eficiente de Markowitz, que representa las carteras que ofrecen el máximo retorno para un nivel dado de riesgo.''')

# Markdown Title
st.markdown('''Paso 7: Graficar la Frontera Eficiente y los Resultados

Graficamos los 10,000 portafolios generados, los puntos de los valores individuales y la frontera eficiente de Markowitz.''')

# Markdown Title
st.markdown('''Paso 8: Crear Gráfico Interactivo Circular con los Pesos de la Cartera Óptima

Finalmente, creamos un gráfico circular para visualizar los pesos de la cartera óptima (con el ratio Sharpe máximo).''')

# Markdown Title
st.markdown('''### Resultados finales modelo 2:''')

# Markdown Title
st.markdown('''# 6.- Carteras optimizadas con Modelos de Machine Learning''')

# Markdown Title
st.markdown('''## 6.1.- Modelo Supervisado ML con Gradient Boosting (Modelo 3)''')

# Markdown Title
st.markdown('''Paso 1: Calcular Retorno y Volatilidad para Cada Valor

Calculamos el retorno anualizado y la volatilidad anualizada para cada uno de los 10 valores. Utilizamos la media y la desviación estándar de los cambios porcentuales diarios (retornos diarios) multiplicados por 252 (el número aproximado de días de trading en un año).''')

# Markdown Title
st.markdown('''Paso 2: Crear la Matriz de Covarianza

La matriz de covarianza mide cómo los retornos de los diferentes valores se mueven juntos. Esta matriz es esencial para calcular la volatilidad de la cartera.''')

# Markdown Title
st.markdown('''Paso 3: Optimización de la Cartera usando Gradient Boosting

Generamos 10,000 combinaciones de pesos al azar para los valores en la cartera y calculamos el retorno esperado, la volatilidad y el ratio Sharpe para cada combinación. 

Luego, entrenamos un modelo de Gradient Boosting para predecir el ratio Sharpe en función de los pesos y utilizamos el modelo para encontrar los pesos que maximizan y minimizan el ratio Sharpe.''')

# Markdown Title
st.markdown('''Paso 4: Mostrar Resultados

Mostramos los resultados de la optimización, incluyendo los pesos de los valores en la cartera, el retorno, la volatilidad y el ratio Sharpe.''')

# Markdown Title
st.markdown('''Paso 5: Frontera Eficiente de Markowitz

Calculamos la frontera eficiente de Markowitz, que representa las carteras que ofrecen el máximo retorno para un nivel dado de riesgo. Utilizamos optimización numérica para encontrar estas carteras.''')

# Markdown Title
st.markdown('''Paso 6: Graficar la Frontera Eficiente y los Resultados

Graficamos la frontera eficiente y los puntos de los valores individuales.''')

# Markdown Title
st.markdown('''Paso 7: Crear Gráfico Interactivo Circular con los Pesos de la Cartera Óptima

Finalmente, creamos un gráfico circular para visualizar los pesos de la cartera óptima (con el ratio Sharpe máximo).''')

# Markdown Title
st.markdown('''### Resultados finales modelo 3:''')

# Markdown Title
st.markdown('''## 6.2.- Modelo Supervisado ML con XGBoost (Modelo 4)''')

# Markdown Title
st.markdown('''Paso 1: Calcular Retorno y Volatilidad para Cada Valor

Primero, calculamos el retorno esperado anualizado y la volatilidad anualizada para cada uno de los 10 valores.''')

# Markdown Title
st.markdown('''Paso 2: Crear la Matriz de Covarianza

La matriz de covarianza mide cómo los retornos de los diferentes valores se mueven juntos. Esta matriz es esencial para calcular la volatilidad de la cartera.''')

# Markdown Title
st.markdown('''Paso 3: Optimización de la Cartera usando XGBoost

Generamos 10,000 combinaciones de pesos al azar para los valores en la cartera y calculamos el retorno esperado, la volatilidad y el ratio Sharpe para cada combinación. 

Luego, entrenamos un modelo de XGBoost para predecir el ratio Sharpe, optimizando posteriormente la cartera.''')

# Markdown Title
st.markdown('''Paso 4: Mostrar Resultados

Mostramos los resultados de la optimización, incluyendo los pesos de los valores en la cartera, el retorno, la volatilidad y el ratio Sharpe.''')

# Markdown Title
st.markdown('''Paso 5: Frontera Eficiente de Markowitz

Calculamos la frontera eficiente de Markowitz, que representa las carteras que ofrecen el máximo retorno para un nivel dado de riesgo. Utilizamos optimización numérica para encontrar estas carteras.''')

# Markdown Title
st.markdown('''Paso 6: Graficar la Frontera Eficiente y los Resultados

Graficamos la frontera eficiente y los puntos de los valores individuales.''')

# Markdown Title
st.markdown('''Paso 7: Crear Gráfico Interactivo Circular con los Pesos de la Cartera Óptima

Finalmente, creamos un gráfico circular para visualizar los pesos de la cartera óptima (con el ratio Sharpe máximo).''')

# Markdown Title
st.markdown('''### Resultados finales modelo 4:''')

# Markdown Title
st.markdown('''## 6.3.- Modelo No Supervisado PCA (Principal Component Analysis) (Modelo 5)''')

# Markdown Title
st.markdown('''Paso 1: Calcular Retorno y Volatilidad para Cada Valor

Primero, necesitamos calcular el retorno esperado anualizado y la volatilidad anualizada para cada uno de los 10 valores.''')

# Markdown Title
st.markdown('''Paso 2: Crear la Matriz de Covarianza

La matriz de covarianza se utiliza para medir cómo los retornos de los valores se mueven juntos. Esta matriz es esencial para calcular la volatilidad de la cartera.''')

# Markdown Title
st.markdown('''Paso 3: Aplicar PCA

Utilizamos PCA para reducir la dimensionalidad y encontrar las componentes principales que expliquen la mayor parte de la variabilidad en los retornos.

 Alineamos los índices de los DataFrames para asegurar que todas las series temporales tengan la misma longitud.''')

# Markdown Title
st.markdown('''Paso 4: Optimización de la Cartera usando las Componentes Principales

Utilizamos las componentes principales para optimizar la cartera.''')

# Markdown Title
st.markdown('''Paso 5: Mostrar Resultados

Mostramos los resultados de la optimización, incluyendo los pesos de los valores en la cartera, el retorno, la volatilidad y el ratio Sharpe.''')

# Markdown Title
st.markdown('''Paso 6: Frontera Eficiente de Markowitz

Calculamos la frontera eficiente de Markowitz, que representa las carteras que ofrecen el máximo retorno para un nivel dado de riesgo.''')

# Markdown Title
st.markdown('''Paso 7: Graficar la Frontera Eficiente y los Resultados

Graficamos la frontera eficiente y los puntos de los valores individuales.''')

# Markdown Title
st.markdown('''Paso 8: Crear Gráfico Interactivo Circular con los Pesos de la Cartera Óptima

Finalmente, creamos un gráfico circular para visualizar los pesos de la cartera óptima (con el ratio Sharpe máximo).''')

# Markdown Title
st.markdown('''### Resultados finales modelo 5:''')

# Markdown Title
st.markdown('''# 7.- Comparacion de resultados finales de los 5 modelos''')

# Markdown Title
st.markdown('''## 7.1.- Comparacion de retorno, volatilidad y sharpe de las 5 carteras''')

# Markdown Title
st.markdown('''Crear la Tabla de Comparación''')

# Markdown Title
st.markdown('''Visualización de resultados''')

# Markdown Title
st.markdown('''Para entender cuál es el mejor modelo, consideramos las tres métricas: retorno, volatilidad y ratio Sharpe. El ratio Sharpe es particularmente importante porque mide el rendimiento ajustado por riesgo de una cartera.

Siendo:

- Retorno: una medida del beneficio esperado de la cartera
- Volatilidad: una medida del riesgo o la variabilidad de los retornos de la cartera
- El ratio Sharpe: mide el rendimiento ajustado por riesgo. Un ratio Sharpe más alto indica un mejor rendimiento ajustado por riesgo''')

# Markdown Title
st.markdown('''Interpretación de los Resultados

1) Modelo 1: Tiene el menor riesgo (volatilidad), pero no el mayor retorno ni el mayor ratio Sharpe.

2) Modelo 2: Ofrece el mayor retorno pero con un riesgo relativamente más alto y un ratio Sharpe menor que el Modelo 3 y 4.

3) Modelo 3: Tiene el mayor ratio Sharpe, lo que indica el mejor rendimiento ajustado por riesgo. Aunque no tiene el mayor retorno, combina un buen retorno con una volatilidad moderada, resultando en el mejor ratio Sharpe.

4) Modelo 4: Tiene un ratio Sharpe muy cercano al del Modelo 3, pero ligeramente inferior.

5) Modelo 5: Tiene el menor retorno y el menor ratio Sharpe, lo que indica un rendimiento inferior ajustado por riesgo en comparación con los otros modelos.''')

# Markdown Title
st.markdown('''Conclusión

El Modelo 3 es el mejor modelo según los resultados, ya que ofrece el mejor rendimiento ajustado por riesgo, como se evidencia por su ratio Sharpe más alto. Aunque el Modelo 2 tiene el mayor retorno, su mayor volatilidad reduce su ratio Sharpe en comparación con el Modelo 3. Por lo tanto, el Modelo 3 proporciona una combinación óptima de retorno y riesgo.''')

# Markdown Title
st.markdown('''## 7.2.- Comparacion de peso de carteras''')

# Markdown Title
st.markdown('''Primero recogemos los datos. Puesto que los modelos 2, 3, 4 y 5 mostraban los resultados de los porcentajes, en formato decimal (20 % = 0.20), van a tener que multiplicarse sus resultados por 100. Con el fin de iguar los resultados con los del modelo 1.''')

# Markdown Title
st.markdown('''Crear la Tabla de Comparación de Pesos

Convertimos los datos de los pesos en un DataFrame para facilitar su manipulación y visualización.''')

# Markdown Title
st.markdown('''Análisis y explicación de resultados

1) Modelo 1: Este modelo ha asignado la mayor parte de la inversión a los índices ASX200 y NIFTY50, con una diversificación significativa en SPCanada. La inversión en bancos (SAN, BBVA) y Nasdaq es nula, sugiriendo una preferencia por índices de mercados más estables.

2) Modelo 2: Similar al Modelo 1, este modelo también favorece NIFTY50 y ASX200, pero incluye una mayor diversificación en REP (8.93%) y BBVA (6.17%). La exposición a bancos y la energía es mayor, con una reducción en la inversión en SP500.

3) Modelo 3: Este modelo presenta una diversificación más equilibrada, incluyendo una inversión notable en Nasdaq (9.69%) y REP (6.02%). La menor inversión en ACS y SP500 sugiere una estrategia de minimización de riesgo en estos sectores.

4) Modelo 4: Este modelo destaca por su alta diversificación y un peso significativo en Nasdaq (12.45%). La inversión en bancos es baja, indicando una posible estrategia de evitar volatilidades asociadas a este sector.

5) Modelo 5: Al igual que los otros modelos, NIFTY50 y ASX200 son los favoritos, pero este modelo muestra una estrategia más conservadora al excluir completamente a BBVA y Nasdaq. La inversión en IBE es elevada (11.68%), indicando una preferencia por este sector.''')

# Markdown Title
st.markdown('''Conclusiones

Cada modelo muestra una estrategia de inversión diferente, pero hay algunas tendencias comunes:

- Favoritismo por NIFTY50 y ASX200: Todos los modelos asignan una gran parte de la inversión a estos índices, indicando una confianza general en estos mercados.
- Diversificación en SPCanada: También es un activo popular en todas las carteras.
- Variabilidad en Activos Específicos: La inversión en bancos y Nasdaq varía significativamente entre los modelos, sugiriendo diferentes enfoques hacia la volatilidad y el riesgo.

El Modelo 3 parece tener el mejor balance de diversificación y riesgo ajustado, como se indicó anteriormente en el análisis del ratio Sharpe. Sin embargo, la elección del modelo final puede depender de las preferencias de riesgo y la estrategia de inversión del inversor.''')

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)

# Plot
fig, ax = plt.subplots()
# Add your plotting code here
st.pyplot(fig)
