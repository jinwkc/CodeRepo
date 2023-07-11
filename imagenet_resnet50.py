import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.distribute import MirroredStrategy,NcclAllReduce
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy,CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import CosineDecay,ExponentialDecay

from CodeRepo.ResNet50 import ResNet50
from CodeRepo.load_data import load_data
import argparse

def args_parser():
    parse=argparse.ArgumentParser()
    parse.add_argument('--batch-size',default=64,type=int)
    parse.add_argument('--epochs', default=300, type=int)
    parse.add_argument('--input-size', default=224, type=int, help='images input size')
    parse.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parse.add_argument('--data-path',default='/home/jovyan/data/imagenet/',type=str)
    
    return parse

if __name__== "__main__":
    
    args=args_parser()
    args=args.parse_args()
    
    print(args.input_size, args.batch_size, args.lr)
    
    # strategy=MirroredStrategy(devices=["GPU:0", "GPU:1"],cross_device_ops=NcclAllReduce(num_packs=2))
    strategy=MirroredStrategy(devices=["GPU:0", "GPU:1","GPU:2","GPU:3"],cross_device_ops=NcclAllReduce(num_packs=2))
    
    # data_path='/home/jovyan/data/imagenet/'
    tr_set,valid_set,test_set=load_data(data_dir_path=args.data_path,img_size=args.input_size,batch_size=args.batch_size)
    
    lr_schedule = CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=1000)
    
    optimizer = Adam(learning_rate = lr_schedule)
    loss = CategoricalCrossentropy(from_logits = True)
    es=EarlyStopping(monitor='val_accuracy', min_delta=0, patience=0, 
                     verbose=0,mode='max', baseline=None, restore_best_weights=False)
    
    with strategy.scope():
        model=ResNet50(input_shape=(args.input_size,args.input_size,3),classes=1000)
        model.compile(optimizer = optimizer,
                      loss = CategoricalCrossentropy(from_logits = True), 
                      metrics = ['accuracy'])

    hist=model.fit(tr_set,epochs=args.epochs,validation_data=valid_set,callbacks=[es])
    
    model.save('ResNet50_224x224_imagenet.h5')
    exit()