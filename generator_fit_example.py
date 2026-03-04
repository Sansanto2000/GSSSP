'''
Docstring for generator_use_example
Se usa el generador SpectrumLabeledSequence para realizar un entrenamiento con Tensorflow.
'''

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # Ocultar mensajes de advertencia
from generators.spectrumLabeledSequence import OutputFormat, SpectrumLabeledSequence
from keras.optimizers import Adam
import tensorflow as tf

MODEL_PATH = '/home/sponte/Repositorios/SpectroscopicObservationDetector/models/model.keras'
DESTINY = '/mnt/data3/sponte/datasets/conGSSSP.test' # "D:\\Datasets\\conGSSSP_v2"
BATCHT_SIZE = 16
LEARNING_RATE = 0.001
GLOBAL_CLIPNORM = 10.0
EPOCH = 2
TRAIN_STEPS = 50
VAL_STEPS = 10

### Generador ###
observations_train_gen = SpectrumLabeledSequence(
    height_range=(500,1400),
    width_range=(500,1400),
    batch_size=BATCHT_SIZE, 
    resize_shape=(640,640), 
    max_predictions=20,
    output_format=OutputFormat.DICT,
    batchs_per_sequence=TRAIN_STEPS
)

observations_val_gen = SpectrumLabeledSequence(
    height_range=(500,1400),
    width_range=(500,1400),
    batch_size=BATCHT_SIZE, 
    resize_shape=(640,640), 
    max_predictions=20,
    output_format=OutputFormat.DICT,
    batchs_per_sequence=VAL_STEPS
)

### Modelo ###
model = tf.keras.models.load_model(
    '/home/sponte/Repositorios/SpectroscopicObservationDetector/models/model.keras',
    compile=False
)
# Optimizador 
optimizer = Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM
)
# Compilar
model.compile(
    optimizer=optimizer,
    classification_loss="binary_crossentropy",
    box_loss='ciou',
    jit_compile=False
)

### Entrenar ###
# dataset_train = tf.data.Dataset.from_generator(
#     lambda: observations_train_gen,
#     output_signature=(
#         tf.TensorSpec(shape=(None, 640, 640, 3), dtype=tf.uint8),
#         {
#             "boxes": tf.RaggedTensorSpec(shape=(None, None, 4), dtype=tf.float32),
#             "classes": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.float32),
#         }
#     )
# )
# dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)
# dataset_val = tf.data.Dataset.from_generator(
#     lambda: observations_val_gen,
#     output_signature=(
#         tf.TensorSpec(shape=(None, 640, 640, 3), dtype=tf.uint8),
#         {
#             "boxes": tf.RaggedTensorSpec(shape=(None, None, 4), dtype=tf.float32),
#             "classes": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.float32),
#         }
#     )
# )
# dataset_val = dataset_val.prefetch(tf.data.AUTOTUNE)

model.fit(
    observations_train_gen,
    #callbacks=callbacks,
    #steps_per_epoch=len(observations_train_gen),
    epochs=EPOCH,
    validation_data=observations_val_gen,
    #validation_steps=len(observations_val_gen)
)