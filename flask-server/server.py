# Import libraries
from flask import Flask,jsonify,request, Response, send_file
from flask_cors import CORS
from Cifar10 import train5
from Cifar100 import train2
from MNIST import train3
from SVHN import train4
from NormalAccCifar10 import normalCifar10
from NormalAccCifar100 import normalCifar100
from NormalAccMNIST import normalMNIST
from NormalAccSVHN import normalSVHN
from SSLSCifar10 import ssls1
from SSLSCifar100 import ssls2
from SSLSMnist import ssls3
from SSLSSvhn import ssls4

from train import train_model
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import json
import os
import io


 
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/run', methods=['GET'])
def runCifar10():
    
    output = train5()
    return jsonify(output=output)

@app.route('/run2', methods=['GET'])
def runCifar100():
   
    output = train2()
    return jsonify(output=output)

@app.route('/run3', methods=['GET'])
def runMNIST():
   
    output = train3()
    return jsonify(output=output)

@app.route('/run4', methods=['GET'])
def runSVHN():
   
    output = train4()
    return jsonify(output=output)




@app.route('/normal1', methods=['GET'])
def normal10():
   
    
    output = normalCifar10()
    return jsonify(output=[str(output)])

@app.route('/normal2', methods=['GET'])
def normal100():
   
    output = normalCifar100()
    return jsonify(output=[str(output)])


@app.route('/normal3', methods=['GET'])
def normalSVHN():
   
    output = normalSVHN()
    return jsonify(output=[str(output)])


@app.route('/normal4', methods=['GET'])
def normalMNIST():
   
    output = normalMNIST()
    return jsonify(output=[str(output)])

@app.route('/ssls1', methods=['GET'])
def new1():
   
    output = ssls1()
    return jsonify(output=[str(output)])

@app.route('/ssls2', methods=['GET'])
def new2():
   
    output = ssls2()
    return jsonify(output=[str(output)])

@app.route('/ssls3', methods=['GET'])
def new3():
   
    output = ssls3()
    return jsonify(output=[str(output)])

@app.route('/ssls4', methods=['GET'])
def new4():
   
    output = ssls4()
    return jsonify(output=[str(output)])
""" @app.route('/uploadFile',methods=['POST'])
def uploadFile():
    if 'file' not in request.files:
        return jsonify({'error' : 'No file uploaded'}),400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error' : 'No file selected'}),400
    try:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        return jsonify({'error' : 'File uploaded successfully'}), 200
    except Exception as e:
        return jsonify({'error' : 'An error occurred while uploading the file: ' + str(e)}), 500 """


@app.route('/train', methods=['POST'])
def train():
    if 'model_file' not in request.files:
        return {'message': 'No model file uploaded.'}
    if 'dataset_name' not in request.form:
        return {'message': 'No dataset name provided.'}
        
    model_file = request.files['model_file']
    dataset_name = request.form['dataset_name']

    # Save the model file to disk
    model_file.save('model.pt')

    # Train the model and get the accuracy
    output = train_model(dataset_name, 'model.pt')
    #os.remove('model.pt')  # remove the saved model file

    return jsonify(output=output)

@app.route('/download_report', methods=['POST'])
def download_report():
    report_path = 'Report.txt'
    return send_file(report_path, as_attachment=True)

with open('Report.txt', 'w') as file:
    file.write('')

    
if __name__ == '__main__':
    app.run(debug=True)