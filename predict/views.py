from django.http import response
from django.shortcuts import render
import pickle
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
# Create your views here.
import requests
from django.views.decorators.csrf import csrf_exempt




@csrf_exempt
def chatbot(request):
    if request.method == 'GET':            
        inp= request.GET.get('inp')
    with open("predict/static/New_intents.json") as file:
        data = json.load(file)
    chat_model = load_model('predict/static/models/estv2.h5')
    # load tokenizer object
    with open('predict/static/pickles/okenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('predict/static/pickles/lbl_encoder.pickle', 'rb') as enc:
        onehot_encoded = pickle.load(enc)
    max_len = 10
    result = chat_model.predict(pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
    print(result)
    print(np.argmax(result))
    tag = onehot_encoded.inverse_transform([np.argmax(result)])
        

    for i in data['intents']:
        if i['tag'] == tag:
            a=np.random.choice(i['responses'])
            return response.JsonResponse(a,safe=False)
# def display(request):

#     return response.JsonResponse(prediction_label,safe=False)
