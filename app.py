from flask import Flask, jsonify, request
import numpy as np
import  torch
import torch.nn as nn
import pandas as pd
import numpy 
from bs4 import BeautifulSoup
from flask_restful import Resource
from collections.abc import Mapping
from torchvision.models import resnet34
from torch.optim import Adam
import librosa
import matplotlib.pyplot as plt
from albumentations import Normalize
from sklearn.preprocessing import normalize
import cv2
from torch import FloatTensor, LongTensor, DoubleTensor




# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


###################################################app

bird_list = ['bewwre', 'osprey', 'orcwar', 'comter', 'houspa', 'blujay', 'redcro', 'wesmea', 'amerob', 'marwre', 'bnhcow', 'norcar', 'daejun', 'chispa', 'cangoo', 'norwat', 'wilfly', 'mallar3', 'whbnut', 'rewbla']
keys = set(bird_list)
values = np.arange(0, len(keys))
code_dict = dict(zip(sorted(keys), values))

class BirdNet(nn.Module):
    def __init__(self, f, o):
        super(BirdNet, self).__init__()
        self.f = f
        self.dropout = nn.Dropout(p=0.2)
        self.dense_output = nn.Linear(f, o)
        self.resnet = resnet34(pretrained=True)
        ct = 0
        for child in self.resnet.children():
            ct += 1
            if ct < 7:  # Freezed first 6 layers 
                for param in child.parameters():
                    param.requires_grad = False

        self.resnet_head = list(self.resnet.children())
        self.resnet_head = nn.Sequential(*self.resnet_head[:-1])
    
    def forward(self, x):
        x = self.resnet_head(x)
        return self.dense_output(self.dropout(x.view(-1, self.f)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BirdNet(f=512,o= 20)
LR = 1e-3, 1e-2
optimizer = Adam([{'params': model.resnet.parameters(), 'lr': LR[0]},
                  {'params': model.dense_output.parameters(), 'lr': LR[1]}])
criterion = torch.nn.CrossEntropyLoss()
model.to(device)



state = torch.load('model_resnet34_epoch_20.pt',map_location= torch.device('cpu'))
model.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('test.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    file = request.files['myfile']
    hope = 512
    y, sr = librosa.load(file,sr = 50000,duration=20)
    y = y + 0.009*np.random.normal(0,1,len(y))

    rmse = librosa.feature.rms(y, hop_length=hope, center=True)
    spectrogram  = librosa.stft(y)
    mfccs  = librosa.feature.mfcc(y, sr=sr,n_mfcc=50)
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hope)
    times = librosa.times_like(oenv, sr=sr, hop_length=hope)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,hop_length=hope)
    chromagram = librosa.feature.chroma_stft(y, sr=sr, hop_length=hope)

    max_len = max([spectrogram_db.shape[0],mfccs.shape[0],tempogram.shape[0],chromagram.shape[0]])
    max_len_c = max([spectrogram_db.shape[1],mfccs.shape[1],tempogram.shape[1],chromagram.shape[1]])

    spectrogram_db = np.pad(spectrogram_db, [(0, max_len-spectrogram_db.shape[0]), (0, max_len_c-spectrogram_db.shape[1])], mode='constant')
    mfccs = np.pad(mfccs, [(0, max_len-mfccs.shape[0]), (0, max_len_c-mfccs.shape[1])], mode='constant')
    tempogram = np.pad(tempogram, [(0, max_len-tempogram.shape[0]), (0, max_len_c-tempogram.shape[1])], mode='constant')
    chromagram = np.pad(chromagram, [(0, max_len-chromagram.shape[0]), (0, max_len_c-chromagram.shape[1])], mode='constant')

    plt.figure(figsize=(30,5))
    final = (spectrogram_db) +  (mfccs) + tempogram + chromagram + (rmse*100)
    final = normalize(final, axis=1, norm='l1')
    librosa.display.specshow(final , sr=sr, x_axis='time',y_axis='log')
    plt.axis('off')
    plt.savefig('test_audio_image.png')
    plt.clf()
    plt.close()
    aug = Normalize(p=1)
    image = cv2.imread('test_audio_image.png')
    image = FloatTensor(aug(image=image)['image'])
    image = image.permute(2,0,1)
    image = image.view(1,image.shape[0],image.shape[1],image.shape[2])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device) # Set model to gpu

    
    with torch.no_grad():
        model.eval();
        inputs = image.to(device) 
        pred = model(inputs)
        

    for key, value in code_dict.items():
        if numpy.argmax(pred.cpu()) == value:
            print('predicted bird is:{} '.format(key))
    

    # return flask.render_template('index.html')
    
    return 'Hare Krishna Hare Krishna Krishna Krishna Hare Hare \n Hare Rama Hare Rama Rama Rama Hare Hare'

if __name__ == '__main__':
    app.run()
