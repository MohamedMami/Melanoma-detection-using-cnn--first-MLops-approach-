import sys
sys.path.append('src/data_processing')
sys.path.append('src/model')
sys.path.append('src/test')
from model_test import model_test
from data_augmentation import data_load
from model_init import model_init
from model_train import model_train
from model_val import model_val

import yaml
with open('confg.yaml',"r") as file :
        params = yaml.safe_load(file)

training_set,test_set,num_classes,class_names = data_load(params)
model = model_init(num_classes)
model , history =  model_train(model,training_set,test_set,params)
model.summary()
print(history.history.keys())
result = model_val(history,test_set,training_set,model)
print("Test-set classification accuracy: {0:.2%}".format(result[1]))
print("Train-set classification accuracy: {0:.2%}".format(result[0]))
model_test(model, class_names)