# Proyecto de Fin de Máster. Explicabilidad en modelos de Machine Learning orientados al Procesamiento del Lenguaje Natural

## Máster Universitario en Ciencia de Datos e Ingeniería de Computadores 21-22

### Descripción del problema

### Objetivos del proyecto

### Datasets

Se pretende utilizar los conjuntos de entrenamiento y validación correspondientes a la competición [*EXIST 2022*](http://nlp.uned.es/exist2022/), cuyo primer propósito consiste en desarrollar modelos de Aprendizaje Automático capaces de **detectar si un texto es sexista o no**. Se trata de la segunda edición a nivel internacional que persigue la creación, diseño y evolución de sistemas de filtrado de contenido que ayuden a mejorar y establecer políticas de igualdad en redes sociales. A continuación se destacan los aspectos más relevantes de sendos datasets:

* Fuentes de datos: **Twitter y Gab**.
* Idiomas: **español e inglés**.
* Número de ejemplos del conjunto de **entrenamiento: 6.977 textos**.
* Número de ejemplos del conjunto de **validación: 4.368 textos**.
* Clases: *sexist*, *non-sexist*. El número de muestras de cada categoría se encuentra prácticamente equilibrado por lo que se determina que las **clases están balanceadas**.
* El número de textos en inglés y en español también es considerablemente similar por lo que, en términos generales, no parece existir **ningún sesgo por idioma** en sendos datasets.
* Columnas "útiles" para el proyecto.
   * `source`: la fuente de datos de la que procede un texto, *twitter* o *gab*.
   * `language`: el idioma en el que se encuentra un texto, *en* o *es*.
   * `text`: el contenido de un texto.
   * `task1`: la etiqueta binario de detección de sexismo, *sexist* o *non-sexist*.
