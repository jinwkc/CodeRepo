from tensorflow.keras.preprocessing.image import ImageDataGenerator


# data_path='/home/jovyan/data/imagenet/'
def load_data(data_dir_path,img_size=64,batch_size=64):
    datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip=True,vertical_flip=True)
                                   # shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, validation_split=0.2)
    training_set = datagen.flow_from_directory(data_dir_path+'train/', target_size = (img_size, img_size),
                                                 batch_size = batch_size, class_mode = 'categorical')
    valid_datagen=ImageDataGenerator(rescale = 1./255,horizontal_flip=True,vertical_flip=True,validation_split=0.4)
    valid_set=valid_datagen.flow_from_directory( data_dir_path+'valid/', target_size = (img_size, img_size),
                                                 batch_size = batch_size, class_mode = 'categorical',subset='training')
    test_set=valid_datagen.flow_from_directory(data_dir_path+'valid/', target_size = (img_size, img_size),
                                                 batch_size = batch_size, class_mode = 'categorical',subset='validation')
    
    return training_set, valid_set, test_set

