# from tensorflow.python.ops import control_flow_ops 
# import sys
# f = open('/dev/null', 'w')
# sys.stdout = f
from flask import Flask, request


app = Flask(__name__)


@app.route('/')
def getFruits():
    return "Fruits!"

@app.route('/model')
def getModel():
	# import sys
	# import os
	# stderr = sys.stderr
	# sys.stderr = open(os.devnull, 'w')
	# from keras.models import load_model
	# sys.stderr = stderr
	# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	import tensorflow as tf
	from SmoothieIt import loadModel
	model = loadModel()
	# print("\n\n\nMODEL"/)
	# model.summary()
	return "MODEL!"

if __name__ == '__main__':
    app.run()