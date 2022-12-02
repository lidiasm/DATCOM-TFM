#!/bin/bash

OUTPUTS_DIR_PATH="outputs"


# 1. Instala la librería 'virtualenv' 
pip install virtualenv==20.17.0
printf "Instalada la librería 'virtualenv' en el sistema.\n"

# 2. Elimina la carpeta de un entorno virtual previamente creado, si existe,
# y crea una nueva con un nuevoentorno virtual
rm -rf venv
virtualenv venv 
printf "Creado un entorno virtual denominado 'venv' en la raíz del repositorio.\n\n"

# 3. Activa el entorno virtual
source venv/bin/activate
printf "Entorno virtual activado.\n\n"

# 4. Instala las dependencias definidas en el fichero 'requirements.txt' dentro del entorno virtual activado
pip install -r requirements.txt
printf "Instaladas las dependencias necesarias en el entorno virtual.\n\n"

# 5. Crea una carpeta 'outputs' para los ficheros y modelos resultantes, si no existe
mkdir -p "$OUTPUTS_DIR_PATH"
printf "El directorio 'outputs' se ha creado en la raíz del repositorio.\n\n"






