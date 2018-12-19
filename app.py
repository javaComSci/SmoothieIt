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
	print("request")
	print(request.form['img1'])
	p1 = ""
	p2 = ""
	p3 = ""
	returnText = "Smoothie recipie requires "
	if 'img1' in request.form:
		p1 = predictNeuralNet(request.form['img1'])
		returnText = returnText + p1
	if 'img2' in request.form:
		p2 = predictNeuralNet(request.form['img2'])
		returnText = returnText + " and " + p2
	if 'img3' in request.form:
		p3 = predictNeuralNet(request.form['img3'])
		returnText = returnText + " and " + p3
	return returnText

if __name__ == '__main__':
    app.run()