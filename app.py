import tensorflow as tf
from tensorflow.python.ops import control_flow_ops 
from flask import Flask, request
from SmoothieIt import loadModel

app = Flask(__name__)


@app.route('/')
def getFruits():
    return "Fruits!"

if __name__ == '__main__':
    model = loadModel()
    print("\n\n\nMODEL")
    model.summary()
    app.run()