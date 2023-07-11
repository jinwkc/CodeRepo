from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

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

def  transflormXY(x,y):
    le=LabelEncoder()
    
    y=le.fit_transform(y)
    y=to_categorical(y)
    x=np.array(x)
    
    return x,y

def load_image_to_path(imagepath_list,image_size=32):
    train_x=[]
    train_y=[]
    valid_x=[]
    valid_y=[]
    test_x=[]
    test_y=[]
    
    for image in imagepath_list:
        try:
            img=cv2.imread(image,cv2.IMREAD_COLOR)
            img=cv2.resize(img,(image_size,image_size))
            
            if 'train' in image:
                train_x.append(np.array(img))
                if 'normal' in image :
                    train_y.append('NORMAL')
                else:
                    train_y.append('POTHOLES')
            elif 'test' in image:
                test_x.append(np.array(img))
                if 'normal' in image :
                    test_y.append('NORMAL')
                else:
                    test_y.append('POTHOLES')
            else:
                valid_x.append(np.array(img))
                if 'normal' in image :
                    valid_y.append('NORMAL')
                else:
                    valid_y.append('POTHOLES')
        except:
            pass
    return transflormXY(train_x,train_y),transflormXY(valid_x,valid_y),transflormXY(test_x,test_y)

def load_data_np_array(data_dir_path,img_size=64):
    imagepath_list=[]
   
    for dirname,_,filenames in os.walk(data_dir_path):
        for filename in filenames:
            path=os.path.join(dirname,filename)
            imagepath_list.append(path)
            
    (train_x,train_y),(valid_x,valid_y),(test_x,test_y)=load_image_to_path(imagepath_list,image_size=img_size)
    return train_x,train_y,valid_x,valid_y,test_x,test_y