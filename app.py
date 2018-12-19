import tensorflow as tf
from tensorflow.python.ops import control_flow_ops 
from flask import Flask, request
from SmoothieIt import loadModel

app = Flask(__name__)


@app.route('/')
def getFruits():
    return "Fruits!"

@app.route('/model')
def getModel():
	model = loadModel()
	print("\n\n\nMODEL")
	model.summary()
	return "MODEL!"

if __name__ == '__main__':
    app.run()