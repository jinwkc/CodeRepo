import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.distribute import MirroredStrategy
from tensorflow.distribute import NcclAllReduce
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import keras_tuner as kt
from kerastuner.tuners import Hyperband
from kerastuner import HyperModel

from ResNet50 import ResNet50

class ResNetHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model=ResNet50(input_shape=self.input_shape,classes=self.num_classes)

        model.compile(
            optimizer=Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['val_accuracy']
        )
        return model
    
def model_builder(hp):
    strategy=MirroredStrategy(devices=["GPU:0", "GPU:1"],cross_device_ops=NcclAllReduce(num_packs=2))
    with strategy.scope():
        model = ResNet50(input_shape=(64,64,3),classes=2)
        hp_learning_rate = hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                metrics = ['accuracy'])
    
    return model