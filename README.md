# Proyecto de Fin de Máster. Explicabilidad en modelos de Machine Learning orientados al Procesamiento del Lenguaje Natural

## Máster Universitario en Ciencia de Datos e Ingeniería de Computadores 21-22

### Descripción del problema

### Objetivos del proyecto

### Datasets

Se pretende utilizar los conjuntos de entrenamiento y validación correspondientes a la **competición** [**EXIST 2022**](http://nlp.uned.es/exist2022/), cuyo primer propósito consiste en desarrollar modelos de Aprendizaje Automático capaces de **detectar si un texto es sexista o no**. Se trata de la segunda edición a nivel internacional que persigue la creación, diseño y evolución de sistemas de filtrado de contenido que ayuden a mejorar y establecer políticas de igualdad en redes sociales. A continuación se destacan los aspectos más relevantes de ambos datasets. Existe un análisis teórico y estadístico en mayor profundidad dentro del notebook [*eda.ipynb*](https://github.com/lidiasm/DATCOM-TFM/blob/main/notebooks/eda.ipynb).

* Fuentes de datos: **Twitter y Gab**.
* Idiomas: **español e inglés**.
* Conjunto de **entrenamiento con 6.977 textos**.
* Conjunto de **validación con 4.368 textos**.
* **Clases binarias balanceadas**: *sexist* y *non-sexist*.

### Propuestas y aproximaciones

La gran mayoría de las soluciones implementadas por los participantes en sendas competiciones se basan en el uso de variantes de **arquitecturas complejas**, tales como *BERT* o *LSTM*. No obstante, existe un amplio número de candidatos que comenzaron experimentando con **algoritmos clásicos** de Aprendizaje Automático como *Random Forest*, *Support Vector Machine* y Regresión Logística. Con el propósito de comprender el problema se establecen ambas aproximaciones como **pilares** fundamentales para la construcción de este proyecto.

### Tecnologías

