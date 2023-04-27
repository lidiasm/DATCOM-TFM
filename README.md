# Proyecto de Fin de Máster. Explicabilidad en modelos de Machine Learning orientados al Procesamiento del Lenguaje Natural

## Máster Universitario en Ciencia de Datos e Ingeniería de Computadores 21-22

### Descripción del problema

### Objetivos del proyecto

* Realizar un análisis exploratorio de los conjuntos de datos *EXIST2021* para comprender las características y patrones del lenguaje sexista en las redes sociales.

* Evaluar y seleccionar técnicas de Procesamiento de Lenguaje Natural adecuadas para el análisis de los mensajes extraídos de redes sociales.

* Desarrollar modelos de Aprendizaje Automático para la detección de mensajes sexistas publicados en redes sociales.

* Evaluar el rendimiento de los modelos anteriores utilizando métricas relevantes de Aprendizaje Automático, como precisión, recall y F1-score.

* Integrar soluciones de explicabilidad e interpretabilidad sobre los modelos con mejor comportamiento para mejorar la extracción de conocimiento y comprensión del problema.

* Realizar un análisis detallado de los resultados para identificar fortalezas y debilidades del modelo y sugerir posibles mejoras.

### Datasets

Se pretende utilizar los conjuntos de entrenamiento y validación correspondientes a la **competición** [**EXIST 2021**](http://nlp.uned.es/exist2021/), cuyo primer propósito consiste en desarrollar modelos de Aprendizaje Automático capaces de **detectar si un texto es sexista o no**. Se trata de la segunda edición a nivel internacional que persigue la creación, diseño y evolución de sistemas de filtrado de contenido que ayuden a mejorar y establecer políticas de igualdad en redes sociales. A continuación se destacan los aspectos más relevantes de ambos datasets. Existe un análisis teórico y estadístico en mayor profundidad dentro del notebook [*eda.ipynb*](https://github.com/lidiasm/DATCOM-TFM/blob/main/notebooks/eda.ipynb).

* Fuentes de datos: **Twitter y Gab**.
* Idiomas: **español e inglés**.
* Conjunto de **entrenamiento con 6.977 textos**.
* Conjunto de **validación con 4.368 textos**.
* **Clases binarias balanceadas**: *sexist* y *non-sexist*.

### Propuestas y aproximaciones

La gran mayoría de las soluciones implementadas por los participantes en sendas competiciones se basan en el uso de variantes de **arquitecturas complejas**, tales como *BERT* o *LSTM*. No obstante, existe un amplio número de candidatos que comenzaron experimentando con **algoritmos clásicos** de Aprendizaje Automático como *Random Forest*, *Support Vector Machine* y Regresión Logística. Con el propósito de comprender el problema se pretende realizar diversas experimentaciones con las siguientes algoritmos clásicos y arquitecturas basadas en texto:

* Regresión Logística: [notebook](https://github.com/lidiasm/DATCOM-TFM/blob/main/notebooks/lr_models.ipynb)

* LSTM y Bi-LSTM

    * [Notebook](https://github.com/lidiasm/DATCOM-TFM/blob/main/notebooks/run_lstm_experiments.ipynb) que ha permitido descubrir la aportación relativa a cada modificación realizada sobre la configuración de los datasets y de los modelos.

    * [Notebook](https://github.com/lidiasm/DATCOM-TFM/blob/main/notebooks/lstm_models.ipynb) con dos modelos de ejemplo, uno para textos ingleses y un segundo para documentos en español, con la mejor configuración identificada para cada caso.

* BERT

    * [Notebook](https://github.com/lidiasm/DATCOM-TFM/blob/main/notebooks/run_bert_experiments.ipynb) que ha permitido descubrir la aportación relativa a cada modificación realizada sobre la configuración de los datasets y de los modelos.

    * [Notebook](https://github.com/lidiasm/DATCOM-TFM/blob/main/notebooks/bert_models.ipynb) con dos modelos de ejemplo, uno para textos ingleses y un segundo para documentos en español, con la mejor configuración identificada para cada caso.


<ins>Nota: debido a restricciones de tamaño los modelos no se encuentran en este repositorio. Pueden contactarme a lidiasm96@correo.ugr.es o lidia.96.sm@gmail.com en caso de que deseen experimentar con ellos.</ins>