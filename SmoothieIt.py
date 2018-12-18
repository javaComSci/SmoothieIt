import tensorflow as tf
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tensorflow import keras
import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def getFlatImg(fileName):
    fruitPics = os.listdir(fileName)
    allFruitPics = []
    for fruitPic in fruitPics:
        img = cv2.imread(os.path.join(fileName, fruitPic))/255.0
        img = img.flatten()
        img = img.reshape((1, img.shape[0]))
        allFruitPics.append(img)   
    allFruitPics = np.vstack(allFruitPics)
    return allFruitPics

def getImg(imgType):
    fruitTypes = os.listdir(imgType)
    fruitTypes = [fruit for fruit in fruitTypes if fruit[0] != '.']
    count = 0
    inputs = []
    outputs = []
    for fruitType in fruitTypes:
        # for the 25 fruit types
        fruitInput = getFlatImg(os.path.join(imgType, fruitType))
        inputs.append(fruitInput)
        fruitOutput = np.ones((fruitInput.shape[0], 1)) * count
        outputs.append(fruitOutput)
        count += 1
    inputs = np.vstack(inputs)
    outputs = np.vstack(outputs)
    outputs = outputs.flatten()
    le_out = LabelEncoder()
    outputs = le_out.fit_transform(outputs) 
    onehot_encoder = OneHotEncoder(sparse=False)
    outputs = outputs.reshape(len(outputs), 1)
    onehot_encoded = onehot_encoder.fit_transform(outputs)
    print("INPUTS",  inputs.shape, inputs)
    print("OUTPUTS", outputs.shape, outputs, onehot_encoded)
    return (inputs, onehot_encoded)
    
def getTrainingImg():
    return getImg('fruits-360/Training/')
    
def getTestingImg():
    return getImg('fruits-360/Testing/')

def createModel(dropoutRates):
	model = Sequential()
	model.add(Dense(50, activation='relu', input_dim=30000))
	model.add(Dropout(dropoutRates[0]))
	model.add(Dense(30, activation='relu'))
	model.add(Dropout(dropoutRates[1]))
	model.add(Dense(25, activation='softmax'))
	model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
	return model

def fitModel(model, inputs, outputs):
	model.fit(inputs, outputs, epochs=6)
	return model

def shuffleData(inputs, outputs):
	n = inputs.shape[0]
	indicies = list(range(0, n))
	np.random.shuffle(indicies)
	return (inputs[indicies], outputs[indicies])
	
def evaluateModel(model, inputs, outputs):
	loss, acc = model.evaluate(inputs, outputs)
	print("Loss", loss, "Accuracy", acc)
	return (loss, acc)

def trainModelWithValidation(trainInput, trainOutput):
	np.random.seed(1234)
	trainInput, trainOutput = shuffleData(trainInput, trainOutput)
	tInput = trainInput[:9000,:]
	tOutput = trainOutput[:9000,:]
	vInput = trainInput[9000:,:]
	vOutput = trainOutput[9000:,:]

	dropoutRates = [[0,0], [.05,.05], [.05,.1], [.1,.15], [.15,.2]]
	accuracies = []
	models = []

	for rate in dropoutRates:
		model = createModel(rate)
		model = fitModel(model, tInput, tOutput)
		loss, accuracy = evaluateModel(model, vInput, vOutput)
		accuracies.append(accuracy)
		models.append(model)

	print("Accuracies", accuracies)
	maxAccuracyIndex = accuracies.index(max(accuracies))
	print("Best dropout rate", dropoutRates[maxAccuracyIndex])
	return models[maxAccuracyIndex]

def testModel(model, testInput, testOutput):
	testLoss, testAcc = evaluateModel(model, testInput, testOutput)
	print("Test loss", testLoss, "test acc", testAcc)

def main():
	# get training and testing input and output
	# training and validation set
	trainInput, trainOutput = getTrainingImg()

	# testing set
	testInput, testOutput = getTestingImg()

	# train and validate to get best model
	model = trainModelWithValidation(trainInput, trainOutput)

	# test on best model
	testModel(model, testInput, testOutput)

if __name__ == "__main__":
	main()