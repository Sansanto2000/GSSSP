# GSSSP

GSSSP (Generator of Synthetic Scans of Spectroscopic Plates) es un conjunto de herramientas para la generacion de imagenes sinteticas de escaneos de placas espectroscopicas.

De cada imagen generada se provee tanto la imagen como la informacion de los elementos que contiene haciendo las imagenes adecuadas para flujos de trabajo con modelos de vision por computadora como YOLO.

![Imagen sintetica de un escaneo de una placa espectroscopica con 2 observaciones.](assets/exampleGeneration3.jpg)

![Imagen sintetica de un escaneo de una placa espectroscopica con 1 observacion. En azul los limites que delimitan la posicion de la observacion generada.](assets/exampleGeneration1.jpg)

![Imagen sintetica de un escaneo de una placa espectroscopica con 4 observaciones. En azul los limites que delimitan la posicion de cada una de las observaciones generadas.](assets/exampleGeneration2.jpg)

## Entorno virtual

Se recomienda usar un entorno virtual para manejar las dependencias de la libraria de generación.

🔨 Crear entorno virtual `.\venv`:

```
python -m venv venv
```

🚀 Activar entorno virtual `.\venv`:

```
.\venv\Scripts\Activate.ps1
```

## Dependencias

📦 Instala las dependencias neceasrias con:

```
pip install -r requirements.txt
```

## Generar

La carpeta contiene un archivo `main.py` que contiene el código experimental para la generación automática de imágenes de observaciones.

```
python -m main
```

Cada imagen producida tiene un archivo de etiquetas con información de los límites de la imagen que definen cada observación individual y los espectros de ciencia y/o de lámparas de comparación que haya en la misma.

## Libreria de generación

`observationArtist.py` encapsula funciones utiles para el dibujado de observaciones en archivos.
