import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import AdamW

from tensorflow.distribute import MirroredStrategy,NcclAllReduce

from CodeRepo.load_data import load_data_np_array
from CodeRepo.ViT import create_vit_classifier_original


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy=MirroredStrategy(devices=["GPU:0", "GPU:1"],cross_device_ops=NcclAllReduce(num_packs=2))
data_path='/home/jovyan/data/classification/'

class CFG:
    num_classes = 2
    input_shape = (32, 32, 3)
    learning_rate = 0.0001
    weight_decay = 0.0001
    batch_size = 256
    num_epochs = 100
    image_size = 32  # We'll resize input images to this size
    patch_size = 6  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [projection_dim * 2, projection_dim,]  
    # Size of the transformer layers
    transformer_layers = 2
    mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

    
def run_experiment(set_strategy=False,strategy=None):
    
    tr_x, tr_y, valid_x, valid_y, test_x, test_y=load_data_np_array(data_dir_path=data_path,img_size=32)
    
    
    optimizer=tf.optimizers.Adam(learning_rate=learning_rate)
    if set_strategy:
        with strategy.scope():
            model=create_vit_classifier_original(tr_x,CFG)
            model.compile(
                optimizer=optimizer,
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[
                    keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                    # keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
                ],
            )
    else:
        model=create_vit_classifier_original(tr_x,CFG)
        model.compile(
                optimizer=optimizer,
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[
                    keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                    # keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
                ],
            )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    # history = model.fit(
    #     x=x_train,
    #     y=y_train,
    #     batch_size=batch_size,
    #     epochs=num_epochs,
    #     validation_split=0.1,
    #     callbacks=[checkpoint_callback],
    # )
    history=model.fit(
    tr,batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
        validation_data=valid,
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history, model

history,model = run_experiment(set_strategy=True,strategy=strategy)


ViT 80

CNN(ResNet) : 85
DeiT(teacher ResNet) : 83 

ResNet Self KD :  85 이상

SDiT : ResNet Self KD 이상
