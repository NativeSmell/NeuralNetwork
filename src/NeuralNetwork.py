import numpy as np
import random
import json
import copy
import time
import progressbar

#import numba

class Activation:
    
    def __init__(self,_type):
                
        self.actFunction = None
        self.difFunction = None
        
        if _type == 'relu':
            self.actFunction = self.relu
            self.difFunction = self.difRelu
            
        elif _type == 'sigmoid':
            self.actFunction = self.sigmoid
            self.difFunction = self.difSigmoid
            
        elif _type == 'softmax':
            self.actFunction = self.softmax
            self.difFunction = self.difSoftmax

        elif _type == 'linear':
            self.actFunction = self.linear
            self.difFunction = self.difLinear
            
        elif _type == 'tangh':
            self.actFunction = self.tangh
            self.difFunction = self.difTangh
            
        elif _type == 'bip_sigmoid':
            self.actFunction = self.bip_sigmoid
            self.difFunction = self.difBip_sigmoid
        
        else: print('Activation not found!:', _type)
        
    #@numba.jit(parallel=True)    
    def activate(self,_in,weights):   
        x = np.dot(_in, weights[:-1]) + weights[-1]  
        return self.actFunction(x)
    
    #@numba.jit(parallel=True)
    def dif(self,x):
        return self.difFunction(x)
    
    #@numba.jit(parallel=True)
    def sigmoid(self,x):
        return 1. / (1. + np.exp(-x))
    
    #@numba.jit(parallel=True)
    def difSigmoid(self,x):
        return (1. - x) * x
                 
    #@numba.jit(parallel=True)
    def softmax(self,x):
        return np.exp(x)
    
    #@numba.jit(parallel=True)
    def difSoftmax(self,x):
        return (1. - x) * x
        
    #@numba.jit(parallel=True)
    def relu(self,x):
        return x if x > 0 else 0.
    
    #@numba.jit(parallel=True)
    def difRelu(self,x):
        return 1. if x > 0 else 0.
       
    #@numba.jit(parallel=True)
    def linear(self,x):
        return x
    
    #@numba.jit(parallel=True)
    def difLinear(self,x):
        return 1.
    
    #@numba.jit(parallel=True)
    def bip_sigmoid(self,x):
        return (2. / (1. + np.exp(-x))) 
    
    #@numba.jit(parallel=True)
    def difBip_sigmoid(self,x):
        return 1./2. * (1. + x) * (1. - x)

    #@numba.jit(parallel=True)
    def tangh(self,x):
        return (np.exp(2.*x) - 1.) / (np.exp(2.*x) + 1.)
    
    #@numba.jit(parallel=True)
    def difTangh(self,x):
        return 1. - out**2.

class Dense:
    
    def __init__(self,size,activation = 'linear', dropoutProp = 0.):
        
        self.size = size
        
        self.dropoutProp = dropoutProp
        
        self.dropout = np.ones(self.size)
        self.updateDropout()
       
        self.output = np.zeros(self.size, dtype = np.float64)
        self.errors = np.array(self.size, dtype = np.float64)
        
        self.weights = None
  
        self.deltaWeights = None
        self.prevDeltaWeights = None
        
        self.inputs = None
        
        self.activation = Activation(activation)
   
    def getSize(self):
        return self.size
    
    #@numba.jit(parallel=True)
    def init(self,input_dim):
        
        self.input_dim = input_dim
        
        self.weights_init()

        self.deltaWeights = np.zeros((self.size, self.input_dim + 1), dtype = np.float64)
        self.prevDeltaWeights = np.zeros((self.size, self.input_dim + 1), dtype = np.float64)
        
        self.inputs = np.zeros(self.input_dim, dtype = np.float64)
        
        
    #@numba.jit(parallel=True)       
    def weights_init(self):
        coeff = 2 / np.sqrt(self.input_dim + 1)
        self.weights = np.random.uniform(-coeff,coeff,self.size * (self.input_dim + 1)).reshape(self.size,self.input_dim + 1) 
   
    #@numba.jit(parallel=True)   
    def forward(self, prevLayerOut, learn = False):
        
        self.output = np.array([self.activation.activate(prevLayerOut, self.weights[j,:]) for j in range(self.size)], dtype = np.float64)
        
        if self.dropoutProp > 0 and learn:
            self.output *= self.dropout
            self.output *= 1 / (1 - self.dropoutProp)
        
        self.inputs = np.append(prevLayerOut,1)
        
        return self.output
    
    #@numba.jit(parallel=True)
    def back(self, errors):
            
        self.errors = np.multiply(errors, np.array([self.activation.dif(self.output[j]) for j in range(self.size)], dtype = np.float64))
       
        if self.dropoutProp > 0:
            self.output *= self.dropout
    
        return np.array([np.dot(self.errors,self.weights[:,j]) for j in range(self.input_dim)], dtype = np.float64)
    
    #@numba.jit(parallel=True)
    def takeDeltaWeights(self, optimizer):
        self.deltaWeights = optimizer.optimFunct(self.inputs,self.errors,self.deltaWeights,self.prevDeltaWeights)
        
    #@numba.jit(parallel=True)
    def correctWeights(self, batchSize):
        
        self.deltaWeights = np.divide(self.deltaWeights,batchSize)
        
        self.weights = np.add(self.weights,self.deltaWeights)
        
        self.prevDeltaWeights = np.copy(self.deltaWeights)
        self.deltaWeights = np.zeros((self.size, self.input_dim + 1), dtype = np.float64)
        
    #@numba.jit(parallel=True)    
    def updateDropout(self):
        if self.dropoutProp > 0:
            self.dropout = np.random.binomial(1,self.dropoutProp,self.size)
        
        else: self.dropout = np.ones(self.size)
        
class Optimizer:
    
    def __init__(self,_type, **params):
                
        self.learningRate = 0.1
        
        if 'learningRate' in params:
            self.learningRate = params['learningRate']
            
        self.momentum = 0.
        
        if 'momentum' in params:
            self.momentum = params['momentum']
        
        if _type == 'backProp':
            self.optimFunct = self.backProp
            
        else: print('Optimizer not found!', _type)
    
    #@numba.jit(parallel=True)
    def optimFunct(self,inputs,errors,deltaWeights,prevDeltaWeights):
        return self.function(inputs,errors,deltaWeights,prevDeltaWeights)
    
    #@numba.jit(parallel=True)
    def changeLR(self,coeff = 2):
        self.learningRate /= coeff
    
    #@numba.jit(parallel=True)        
    def backProp(self,inputs,errors,deltaWeights,prevDeltaWeights):
              
        grad = np.array([np.multiply(inputs,errors[i]) for i in range(len(errors))], dtype = np.float64)  
        
        delta = np.multiply(grad,self.learningRate)
        
        if self.momentum > 0:
            momentums = np.multiply(prevDeltaWeights,self.momentum)
            delta = np.add(delta,momentums)
        
        return np.add(deltaWeights,delta)
    
class Metric:
    
    def __init__(self,_type):
        
        self.metric = None
        
        if _type == 'MSE':
            self.metric = self.MSE
        
        elif _type == 'sqrt_MSE':
            self.metric = self.sqrt_MSE
        
        elif _type == 'arctan':
            self.metric = self.arctan
     
        else : print('Metric not found!' , _type)
    
    #@numba.jit(parallel=True)    
    def cal_loss(self, ideal, answer):
        return self.metric(ideal,answer)
            
    #@numba.jit(parallel=True)
    def MSE(self, ideal, answer):
        x = (ideal - answer) ** 2
        return np.sum(x) / len(x)
    
    #@numba.jit(parallel=True)
    def sqrt_MSE(self,ideal,answer):
        x = (ideal - answer) ** 2
        return np.sqrt(np.sum(x) / len(x))
    
    #@numba.jit(parallel=True)
    def arctan(self,ideal,answer):
        x = np.arctan((ideal - answer)**2)
        return np.sum(x) / len(x)
    
class Sequental:
    
    def __init__(self, input_dim):
        
        self.input_dim = input_dim
        self.layers = []
        
        self.size = 0

        self.optimizer = None
        self.metric = None
        
    #@numba.jit(parallel=True)
    def add(self,layer):
        
        input_dim = self.input_dim if self.size == 0 else self.layers[-1].getSize()
        
        layer.init(input_dim)
 
        self.layers.append(layer)
        self.size += 1
        
    #@numba.jit(parallel=True)
    def predict(self,example, learn = False):
        
        signal = example
        
        for layer in range(self.size):
            signal = self.layers[layer].forward(signal, learn = learn)
            
        return signal
    
    #@numba.jit(parallel=True)
    def errorBack(self,ideal,answer):
        error = ideal - answer
        for layer in range(self.size -1, -1, -1):
            error = self.layers[layer].back(error)
            
    
    #@numba.jit(parallel=True)
    def shuffle(self,A,B):

        data = list(zip(A, B))
        random.shuffle(data)

        return zip(*data)
    
          
    #@numba.jit(parallel=True)
    def learning(self, train_X, train_Y,batchSize):
            
        bar_counter = 0
        batchCounter = 0
        
        bar = progressbar.ProgressBar(max_value = len(train_X) / batchSize)
        time.sleep(0.5)
        
        for i in range(len(train_X)):
                
            answer = self.predict(np.array(train_X[i], dtype = np.float64) , learn = True)
            ideal = np.array([1 if train_Y[i] == j else 0 for j in range(len(answer))], dtype = np.float64)
            
            self.errorBack(ideal,answer)
            
            for layer in self.layers:
                layer.takeDeltaWeights(self.optimizer)
            
            batchCounter += 1
            
            if batchCounter == batchSize:
    
                for layer in self.layers:
                    layer.correctWeights(batchSize)
                    layer.updateDropout()
                    
                batchCounter = 0
                
                bar.update(bar_counter)
                bar_counter += 1
        
        bar.finish()
    
    #@numba.jit(parallel=True)
    def evaluate(self,X,Y):
        
        acc = 0
        loss = []
        
        print("Evaluating...")
        
        bar = progressbar.ProgressBar(max_value =len(X))
        time.sleep(0.5)
        
        bar_counter = 0
        for i in range(len(X)):
        
            answer = self.predict(np.array(X[i], dtype = np.float64) , learn = False)
            ideal = np.array([1 if Y[i] == j else 0 for j in range(len(answer))], dtype = np.float64)
            
            loss.append(self.metric.cal_loss(ideal,answer))
    
            if np.argmax(answer) == Y[i]:
                acc += 1
            
            bar.update(bar_counter)    
            bar_counter += 1
            
        return acc / len(X) , sum(loss) / len(loss)
        
    
    
    def fit(self, train_X, train_Y, val_X, val_Y, numOfEpochs = 100, earlyStopAcc = 0.001,lossMetric = 'MSE', optimizerType = 'backProp', batchSize = 1, learningRate = 0.1, momentum = 0.9, patience = 3):
        
        self.optimizer = Optimizer(optimizerType, learningRate = learningRate, momentum = momentum)
        self.metric = Metric(lossMetric)
        
        history = {
            
            'loss' : [],
            'accuracy' : [],
            'val_loss' : [],
            'val_accuracy' : [],
            'lossMetric' : lossMetric,
            'optimizer' : optimizerType,
            'batchSize': batchSize,
            'earlyStopAcc' : earlyStopAcc,
            'learningRate' : learningRate,
            'patience' : patience,
            'momentum' : momentum
            
        }
        print('Start Learning...')

        prevLoss = 0
        patienceCounter = 0

        for epoch in range(numOfEpochs):

            train_X, train_Y = self.shuffle(train_X,train_Y)

            self.learning(train_X,train_Y,batchSize)

            acc, loss = self.evaluate(train_X,train_Y)
            history['accuracy'].append(acc)
            history['loss'].append(loss)
           
            acc, loss = self.evaluate(val_X,val_Y)
            history['val_accuracy'].append(acc)
            history['val_loss'].append(loss)
            val_acc = ' - val_loss(' + lossMetric + '): ' + str(history['val_loss'][epoch]) + ' - val_accuracy: ' + str(history['val_accuracy'][epoch])

            print('Epoch: ' + str(epoch + 1) + '/' + str(numOfEpochs) + ' - loss(' + lossMetric + '): ' + str(history['loss'][epoch]) + ' - accuracy: ' + str(history['accuracy'][epoch]) + val_acc)

            if abs(loss - prevLoss) < earlyStopAcc:
                self.optimizer.changeLR(coeff = 2)
                patienceCounter += 1

            else:
                patienceCounter = 0

            prevLoss = loss 

            if patienceCounter >= patience:
                break

        return history
