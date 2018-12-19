# from tensorflow.python.ops import control_flow_ops 
# import sys
# f = open('/dev/null', 'w')
# sys.stdout = f
from flask import Flask, request, render_template
print("__name__", __name__)
app = Flask(__name__)


@app.route('/')
def getFruits():
	return render_template("smoothieit/public/index.html")
    # return "Fruits!"

@app.route('/App.js')
def getApp():
	return render_template("smoothieit/public/App.js")

@app.route('/Fruit.js')
def getFruit():
	return render_template("smoothieit/public/Fruit.js")

@app.route('/model', methods = ['POST'])
def getModel():
	# import sys
	# import os
	# stderr = sys.stderr
	# sys.stderr = open(os.devnull, 'w')
	# from keras.models import load_model
	# sys.stderr = stderr
	# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	import tensorflow as tf
	from SmoothieIt import predictNeuralNet
	model = predictNeuralNet()
	model.evaluate
	# print("\n\n\nMODEL"/)
	# model.summary()
	return "MODEL!"

if __name__ == '__main__':
    app.run()