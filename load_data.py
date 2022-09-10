from tensorflow.keras.preprocessing.image import ImageDataGenerator


# data_path='/home/jovyan/data/imagenet/'
def load_data(data_dir_path,img_size=64,batch_size=64,class_mode='categorical'):
    datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip=True,vertical_flip=True)
                                   # shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, validation_split=0.2)
    training_set = datagen.flow_from_directory(data_dir_path+'train/', target_size = (img_size, img_size),
                                                 batch_size = batch_size, class_mode = class_mode)
    # valid_datagen=ImageDataGenerator(rescale = 1./255,horizontal_flip=True,vertical_flip=True,validation_split=0.4)
    valid_set=datagen.flow_from_directory( data_dir_path+'valid/', target_size = (img_size, img_size),
                                                 batch_size = batch_size, class_mode = class_mode)
    test_set=datagen.flow_from_directory(data_dir_path+'test/', target_size = (img_size, img_size),
                                                 batch_size = batch_size, class_mode = class_mode)
    
    return training_set, valid_set, test_set

