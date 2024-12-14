from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

import os 
def data_load(params):        
    train_datagen = ImageDataGenerator(rescale = 1.0/255,shear_range=params['shear_range'],zoom_range=params['zoom_range'],horizontal_flip=params['horizontal_flip'])
    test_datagen = ImageDataGenerator(rescale=params['rescale'])
    
    if os.path.exists(os.getcwd() + params['testdirectoryAug']):
        shutil.rmtree(os.getcwd() + params['testdirectoryAug'])
    os.makedirs(os.getcwd() + params['testdirectoryAug'], exist_ok=True)

    if os.path.exists(os.getcwd() + params['traindirectoryAug']):
        shutil.rmtree(os.getcwd() + params['traindirectoryAug'])
    os.makedirs(os.getcwd() + params['traindirectoryAug'], exist_ok=True)  
    
    training_set= train_datagen.flow_from_directory('Data/origin/training_set',target_size=(124,124),batch_size=16,class_mode='categorical',save_to_dir=os.getcwd()+params['traindirectoryAug'],save_format='jpeg')
    test_set = test_datagen.flow_from_directory('Data/origin/test_set',target_size=(124,124),batch_size=16,class_mode='categorical',save_to_dir=os.getcwd()+params['testdirectoryAug'],save_format='jpeg')
    print(len(training_set))
    num_classes = training_set.num_classes
    class_names = list(training_set.class_indices.keys())
    
    return training_set,test_set,num_classes,class_names
        
    

