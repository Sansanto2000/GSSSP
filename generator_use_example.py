'''
Docstring for generator_use_example
Se crea un lote de imagenes con el generador SpectrumLabeledSequence 
compatible con Tensorflow.
'''

import os
import cv2
from tqdm import tqdm
from lib.observationArtist import drawObservation, labelDictToYolov11Format, labelListToYolov11Format
import numpy as np
from generators.spectrumLabeledSequence import SpectrumLabeledSequence

destiny = "D:\\Datasets\\conGSSSP_v2"

spectrum_gen = SpectrumLabeledSequence(
    height_range=(500,1500),
    width_range=(500,1500),
    batch_size=2, 
    resize_shape=(640,640), 
    max_predictions=20
)

### Guardar elementos de la cantidad de lostes indicados ###
batch_cant = 2
i = 0
for batch_nro in tqdm(range(batch_cant)):
    batch_x, batch_y = spectrum_gen[i]

    for x, y in zip(batch_x, batch_y):

        # Guardar imagen sintetica
        filepath = os.path.join(destiny,"images",f"{i}.jpg")
        success = cv2.imwrite(filepath, x)

        if not success:
            print("¡Error al guardar la imagen! Verifica la ruta y permisos.")
        
        # Convertir etiquetas a formato Yolov11
        filtered = y[~np.all(y == 0, axis=1)]
        lines = map(labelListToYolov11Format, filtered)

        # Guardar etiquetas
        filepath = os.path.join(destiny,"labels",f"{i}.txt")
        if lines:
            with open(filepath, "w") as f:
                f.write("\n".join(lines))
        else:
            # Recomendado: no crear archivo si no hay objetos
            if os.path.exists('img.txt'):
                os.remove('img.txt')

        # Incrementar contador
        i += 1
