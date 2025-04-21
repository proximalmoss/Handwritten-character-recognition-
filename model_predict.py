from keras.models import load_model
from utils import word_dict,preprocess_image
#loading the model
model=load_model('model_hand.h5')
#funtion for predicting letter
def predict_letter(img):
    processed=preprocess_image(img)
    prediction=model.predict(processed)
    result=word_dict[prediction.argmax()]
    return result