import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.distribute import MirroredStrategy,NcclAllReduce
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy,CategoricalCrossentropy

import kerastuner as kt
from kerastuner.tuners import Hyperband
from kerastuner import HyperModel
from kerastuner import HyperParameters
from CodeRepo.ResNet50 import ResNet50

strategy=MirroredStrategy(devices=["GPU:0", "GPU:1"],cross_device_ops=NcclAllReduce(num_packs=2))
data_path='/home/jovyan/data/classification/'

def load_data(data_dir_path,img_size=64,batch_size=64):
    datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip=True,vertical_flip=True)
                                   # shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, validation_split=0.2)
    training_set = datagen.flow_from_directory(data_dir_path+'train/', target_size = (img_size, img_size),
                                                 batch_size = batch_size, class_mode = 'categorical')
    valid_set=datagen.flow_from_directory(data_dir_path+'valid/', target_size = (img_size, img_size),
                                                 batch_size = batch_size, class_mode = 'categorical')
    test_set=datagen.flow_from_directory(data_dir_path+'test/', target_size = (img_size, img_size),
                                                 batch_size = batch_size, class_mode = 'categorical')
    
    return training_set, valid_set, test_set

tr_set,valid_set,test_set=load_data(data_dir_path=data_path,img_size=128,batch_size=128)

def model_builder(hp):
    
    
    # model.summary()
    # Tune the learning rate for the optimizer 
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    with strategy.scope():
        model=ResNet50(input_shape=(128,128,3),classes=2)
        hp_learning_rate = hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
        model.compile(optimizer = Adam(learning_rate = hp_learning_rate),
                loss = CategoricalCrossentropy(from_logits = True), 
                metrics = ['accuracy'])
    
    return model

tuner = Hyperband(
    model_builder,
    # hypermodel,
    max_epochs=1000,
    objective='val_accuracy',
    seed=1,
    executions_per_trial=2,
    directory='./data/hyperband',
    project_name='resnet'
)

tuner.search(tr_set, epochs=1000,validation_data =valid_set)

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
best_model = tuner.get_best_models(num_models=1)[0]
print(f"""
The hyperparameter search is complete. The optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

best_model.save('ResNet50_128x128_Fin_Optimize.h5')
