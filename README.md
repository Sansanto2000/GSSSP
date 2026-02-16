# GSSSP

GSSSP (Generator of Synthetic Scans of Spectroscopic Plates) es un conjunto de herramientas para la generación de imágenes sintéticas de escaneos de placas espectroscópicas.

De cada imagen generada se provee tanto la imagen como la información de los elementos que contiene haciendo las imágenes adecuadas para flujos de trabajo con modelos de visión por computadora como YOLO.

![Imagen sintética de un escaneo de una placa espectroscópica con 2 observaciones.](assets/exampleGeneration3.jpg)

![Imagen sintética de un escaneo de una placa espectroscópica con 1 observación. En azul los limites que delimitan la posición de la observación generada.](assets/exampleGeneration1.jpg)

![Imagen sintética de un escaneo de una placa espectroscópica con 4 observaciones. En azul los limites que delimitan la posición de cada una de las observaciones generadas.](assets/exampleGeneration2.jpg)

## Entorno virtual

Se recomienda usar un entorno virtual para manejar las dependencias de la libreria de generación.

🔨 Crear entorno virtual `.\venv`:

```
python -m venv venv
```

🚀 Activar entorno virtual `.\venv`:

```
# Windows
.\venv\Scripts\Activate.ps1

# Linux
source venv/bin/activate

# Mac
source venv/bin/activate
```


## Dependencias

📦 Instala las dependencias necesarias con:

```
pip install -r requirements.txt
```

## Generar

La carpeta contiene un archivo `main.py` que contiene el código experimental para la generación automática de imágenes de observaciones.

```
python -m main
```

Cada imagen producida tiene un archivo de etiquetas con información de los límites de la imagen que definen cada observación individual y los espectros de ciencia y/o de lámparas de comparación que haya en la misma.

### Compatible con TensorFlow.

En `generators\spectrumLabeledSequence` se encuentra un generador compatible con la librería TensorFlow. El archivo `generator_use_example.py` muestra un ejemplo de como usarla para generar y almacenar archivos, este puede ser usado como se muestra a continuación. 

```
python -m generator_use_example
```

Su propósito es ser usada como alimentador dentro de la función **fit()** de TensorFlow.

Los datos generados por la misma siempre son redimensionados a una dimensión objetivo (se puede especificar). No obstante, si se quieren imágenes sin redimensionar la opción anterior es la correcta.


## Librería de generación

`observationArtist.py` encapsula funciones útiles para el dibujado de observaciones en archivos.
