from keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks import EarlyStopping
import pickle



def model_train(model,training_set,test_set,params):
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.compile(optimizer = params['optimizer'], loss = params['Loss'], metrics = ['accuracy'])
    history= model.fit(training_set,epochs = params['epochs'],validation_data = test_set)#callbacks=[early_stopping]
    with open('Models\model.pkl', 'wb') as f:
        pickle.dump(model.to_json(), f)
    return model , history 