from keras.utils import Sequence
import numpy as np
from lib.observationArtist import drawObservation, add_realistic_noise, labelDictToYolov11Format
import random
import numpy as np
import math
from numbers import Number
import cv2

"""Generador compatible con tensorflow para alimentar modelos 
de aprendizaje automatico. 
"""
class SpectrumLabeledSequence(Sequence):

  """Inicializador.
  
  Params:
  - height_range: rango de configuracion aleatoria para la altura del canvas
  - width_range: rango de configuracion aleatoria para el ancho del canvas
  - gray_value_range: rango porcentual de nivel de gris del fondo. 0 es negro, 1 es blanco.
  - angle_range: rango en grados de angulo de inclinacion de las observaciones.
  - opening_lamp_range: rango porcentual de que tan anchos seran los espectros de lampara en 
  relacion a los especros de ciencia.
  - distance_between_components_range: rango porcentual del espacio vacio entre cada 
  componente de la observacion.
  - distance_between_observations_range: rango porcentual del espacio vacio entre cada 
  observacion.
  - cant_observations_max: numero de observaciones maximo.
  - noise_horizontal: porcentaje de desviacion horizontal para el centro de una observacion.
  - noise_vertical: porcentaje de desviacion horizontal para el centro de una observacion.
  - gaussian_std_range: rango de ruido gaussiano general par la imagen generada.
  - band_intensity_range: rango de ruido de banda horizontal.
  - speck_count_range: rango de cantidad de manchas de polvo.
  - speck_size_range: rango de tamaño de las manchas de polvo.
  - blur_kernel_size_options: lista de opciones enteras para el tamaño del kernel de 
  desenfoque.
  - batch_size: cantidad de elementos por lote.
  - resize_shape: dimensiones (ancho, alto) para las imagenes finales.
  - max_predictions: cantidad maxima de predicciones que puede haber en una imagen.
  """
  def __init__(
      self, *, 
      height_range = (1000, 4000), 
      width_range = (1000, 4000), 
      gray_value_range = (0, 0.15), 
      angle_range = (-5, 5), 
      opening_lamp_range = (0.1, 0.35), 
      distance_between_components_range = (0.02, 0.1),
      distance_between_observations_range = (0.1, 0.8), 
      cant_observations_max = 5,
      noise_horizontal = 0.01, 
      noise_vertical = 0.01, 
      gaussian_std_range= (4.0, 16.0), 
      band_intensity_range = (0.0, 1.0), 
      speck_count_range = (0, 10), 
      speck_size_range = (1,5), 
      blur_kernel_size_options = [1, 3, 5, 7, 9, 11, 13, 15],
      batch_size = 128,
      resize_shape = (640, 640),
      max_predictions = 50
    ):
    
    self.height_range = height_range
    self.width_range = width_range
    self.gray_value_range = gray_value_range
    self.angle_range = angle_range
    self.opening_lamp_range = opening_lamp_range
    self.distance_between_components_range =distance_between_components_range
    self.distance_between_observations_range = distance_between_observations_range
    self.cant_observations_max = cant_observations_max
    self.noise_horizontal = noise_horizontal
    self.noise_vertical = noise_vertical
    self.gaussian_std_range = gaussian_std_range
    self.band_intensity_range = band_intensity_range
    self.speck_count_range = speck_count_range
    self.speck_size_range = speck_size_range
    self.blur_kernel_size_options = blur_kernel_size_options
    self.batch_size = batch_size
    self.resize_shape = resize_shape
    self.max_predictions = max_predictions

  # Number of batch in the Sequence.
  def __len__(self):
    #return math.ceil((self.max_index - (self.min_index + self.lookback)) / self.batch_size)
    return 500

  # Obtener el lote numero idx
  def __getitem__(self, idx):

    batch_x = []
    batch_y = []
    
    for i in range(self.batch_size):
      ### Canvas ###
      # Dimensiones.
      alto = random.randint(*self.height_range)
      ancho = random.randint(*self.width_range)
      # Imagen base oscura completa.
      gray_value = np.random.randint(self.gray_value_range[0]*255, self.gray_value_range[1]*255)
      img = np.full((alto, ancho, 3), gray_value, dtype=np.uint8)

      ### Observacion ###
      # Ancho de la observacion que varia en relacion al ancho total disponible.
      obs_width = random.randint(int(ancho*0.4), int(ancho*0.99))
      # Alto total de la observacion que varia en relacion al ancho de la misma.
      obs_heigth = random.randint(int(obs_width*0.1), int(obs_width*0.4))
      # Inclinacion de la observacion.
      angle = random.randint(*self.angle_range)
      # Que tan anchas van a ser los espectros de lampara en relacion al espectro de ciencia
      openingLamp = random.uniform(*self.opening_lamp_range)
      # Cuanto espacio vacio hay entre cada lampara y el espectro de ciencia.
      distanceBetweenParts = random.uniform(*self.distance_between_components_range)

      ### Grupo de observaciones ###
      # Distancia entre distintas observaciones
      distanceBetweenObservations = random.uniform(
        obs_heigth*self.distance_between_observations_range[0], 
        obs_heigth*self.distance_between_observations_range[1]
      )
      # Cantidad de observaciones que entran en la imagen
      max_observations = math.floor(alto*0.9/(obs_heigth+distanceBetweenObservations))
      # Cuantas observaciones se dibujaran en una la imagen
      n_observations = min(max_observations, random.randint(1, self.cant_observations_max+1))

      ### Definir posiciones ###
      # Posiciones donde ser realizara el dibujo centradas en alto
      noise_horizontal = self.noise_horizontal # Irregularidad porcentual horizontal maxima
      noise_vertical = self.noise_vertical # Irregularidad porcentual veartical maxima
      unit = obs_heigth + distanceBetweenObservations# Espacio a considerar por observación
      targets = []
      for i in range(n_observations):
        pos_y = (alto/2) - (n_observations/2)*unit + unit/2 + i*unit
        coor = {
          "x": ancho/2 + random.uniform(-noise_horizontal, noise_horizontal), 
          "y": pos_y + random.uniform(-noise_vertical, noise_vertical), 
        }
        targets.append(coor)

      ### Dibujar ###
      labels = []
      for coor in targets:
        img, _obs, _mask, label = drawObservation(
          img=img,
          x=coor["x"], 
          y=coor["y"],
          width=int(obs_width),
          height=int(obs_heigth),
          opening=openingLamp,
          distanceBetweenParts=distanceBetweenParts,
          angle=angle,
          baseGrey=gray_value,
          inplace=True,
          debug=False,
        )
        labels.append(label)

      ### Ruido y manchas ###
      # Ruido gaussiano general para la imagen de la placa
      gaussian_std = random.uniform(*self.gaussian_std_range)
      # Ruido de banda horizontal
      band_intensity = random.uniform(*self.band_intensity_range)
      # Cantidad de manchas de polvo
      speck_count = random.randint(*self.speck_count_range)
      # Radio maximo de las manchas de polvo
      speck_size = random.randint(*self.speck_size_range)
      # Nivel del desenfoque gaussiano
      blur_kernel_size = random.choice(self.blur_kernel_size_options)
      # Añadir ruido en la imagen
      img = add_realistic_noise(
        img, 
        gaussian_std=gaussian_std,
        band_intensity=band_intensity,
        speck_count=speck_count,
        speck_size=speck_size,
        blur_ksize=blur_kernel_size,
      )

      # 1. Normalización de la imagen para YOLO
      #img_normalized = img.astype('float32') / 255.0
      # 2. Redimensionar imagen
      img = cv2.resize(img, (self.resize_shape[0], self.resize_shape[1]))
      batch_x.append(img)
      
      # Ajustar dimension maxima etiquetas
      labels_padded = np.zeros((self.max_predictions, 5), dtype=np.float32)
      for i, label in enumerate(labels):
          labels_padded[i] = [
            label['class_id'], 
            label['x_center_norm'], 
            label['y_center_norm'], 
            label['width_norm'], 
            label['height_norm']
          ]
      batch_y.append(labels_padded)

    # Imagenes
    samples = np.array(batch_x)

    # Arreglo de etiquetas de cada imagen
    targets =np.array(batch_y)

    return samples, targets