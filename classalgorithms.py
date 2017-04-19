from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math
class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        
    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
            
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
        

class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    # TODO: implement learn and predict functions                  


    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        #yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        reg = 'l2'
        regularizationVal = 0.2
        regularizationVal2 = 0.01
        numsamples = Xtrain.shape[0]
        initialWeights = np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T, Xtrain)), Xtrain.T),yt)
        self.weights = initialWeights
        gradient = None
        currentPrediction = np.zeros(self.weights.shape[0])
        negetiveHessianInverse = None
        tolerance = 1
        while tolerance > 7.144378975398573e-7:
            prevPreduction = currentPrediction
            currentPrediction = self.predictnum(Xtrain)
            tolerance = 0
            for index in range(prevPreduction.shape[0]):
                tolerance += math.pow(prevPreduction[index] - currentPrediction[index],2)
            tolerance = math.sqrt(tolerance)
            print(tolerance)
            if reg == 'false':
                gradient = np.dot(Xtrain.T, np.subtract(ytrain,currentPrediction ))
                negetiveHessianInverse = np.linalg.pinv(np.dot(np.dot(np.dot(Xtrain.T, np.diag(currentPrediction)),
                                                                  np.subtract(np.identity(currentPrediction.shape[0]) ,np.diag(currentPrediction) )),Xtrain))
            #l2 regularizer
            if reg == 'l2':
                #l2
                gradient = np.add(np.dot(Xtrain.T, np.subtract(ytrain, currentPrediction)), 2 * regularizationVal * self.weights)
                regHessianComponent = np.zeros(self.weights.shape[0])
                regHessianComponent.fill(2*regularizationVal)
                negetiveHessianInverse = np.linalg.pinv(np.add(np.dot(np.dot(np.dot(Xtrain.T, np.diag(currentPrediction)),
                                                                      np.subtract(np.identity(currentPrediction.shape[0]),
                                                                        np.diag(currentPrediction))), Xtrain),0.4*regHessianComponent))
                self.weights = np.add(self.weights,0.5 * gradient)
                continue
            #l1 regularizer
            if reg == 'l1':
                #l1
                gradient = np.add(np.dot(Xtrain.T, np.subtract(ytrain, currentPrediction)), regularizationVal* utils.dl1(self.weights))
                regHessianComponent = np.zeros(self.weights.shape[0])
                regHessianComponent.fill(regularizationVal)
                negetiveHessianInverse = np.linalg.pinv(np.dot(np.dot(np.dot(Xtrain.T, np.diag(currentPrediction)),
                                                                      np.subtract(
                                                                          np.identity(currentPrediction.shape[0]),
                                                                          np.diag(currentPrediction))), Xtrain))

            #l2 regularizer
            if reg == 'elastic':
                #elastic
                gradient = np.add(np.add(np.dot(Xtrain.T, np.subtract(ytrain, currentPrediction)), 2 * regularizationVal * self.weights),regularizationVal2* utils.dl1(self.weights))
                regHessianComponent = np.zeros(self.weights.shape[0])
                regHessianComponent.fill(2*regularizationVal)
                negetiveHessianInverse = np.linalg.pinv(np.add(np.dot(np.dot(np.dot(Xtrain.T, np.diag(currentPrediction)),
                                                                      np.subtract(np.identity(currentPrediction.shape[0]),
                                                                        np.diag(currentPrediction))), Xtrain),0.4*regHessianComponent))



            self.weights = np.add(self.weights,np.dot(negetiveHessianInverse,gradient))

    def predictnum(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        # for i in range(ytest.shape[0]):
        #     ytest[i] = 1/(1+math.exp(-ytest[i]))
        return utils.sigmoid(ytest)

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest = utils.sigmoid(ytest)
        # for i in range(ytest.shape[0]):
        #     if (1/(1+math.exp(-ytest[i]))) >= 0.5:
        #         ytest[i] = 1
        #     else:
        #         ytest[i] = 0
        ytest[ytest >= 0.5] =1
        ytest[ytest < 0.5] = 0
        return ytest

class NaiveBayes(Classifier) :
    def __init__(self,parameters={'usecolumnones':False}):
        self.params = parameters
        self.meanListOne = []
        self.meanListZero = []
        self.stdDevListOne = []
        self.stdDevListZero = []
        self.probOne = 0
        self.reset(parameters)

    def reset(self,params):
        self.resetparams(params)

    def learn(self, Xtrain, ytrain):
        XtrainData = Xtrain
        if not self.params['usecolumnones'] :
            XtrainData = Xtrain[:,:-1]

        numSamples = XtrainData.shape[0]
        positiveIndices = ytrain ==1

        for i in range(XtrainData.shape[1]):
            features = XtrainData[:,i]
            self.meanListOne.append(utils.mean(features[positiveIndices]))
            self.stdDevListOne.append(utils.stdev(features[positiveIndices]))
            self.meanListZero.append(utils.mean(features[~positiveIndices]))
            self.stdDevListZero.append(utils.stdev(features[~positiveIndices]))
        self.probOne = utils.mean(ytrain)

    def predict(self, Xtest):

        predictedValues = []
        XtestData = Xtest
        if not self.params['usecolumnones'] :
            XtestData = Xtest[:,:-1]
        numSamples = XtestData.shape[0]
        for i in range(numSamples):
            dataPoint = XtestData[i]
            pOne = self.probOne
            pZero = 1- pOne

            for j in range(len(dataPoint)):
                pOne = pOne * utils.calculateprob(dataPoint[j],self.meanListOne[j],self.stdDevListOne[j])
                pZero = pZero * utils.calculateprob(dataPoint[j],self.meanListZero[j],self.stdDevListZero[j])

            if pOne > pZero :
                predictedValues.append(1)
            else  :
                predictedValues.append(0)
        return predictedValues

class NeuralNet(Classifier):

    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                        'transfer': 'sigmoid',
                        'stepsize': 0.01,
                        'epochs': 10}
        self.reset(parameters)        

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')      
        self.wi = None
        self.wo = None

    def learn(self, Xtrain, ytrain):

        self.ni = Xtrain.shape[1]
        self.numberOfFeatures = Xtrain.shape[1]
        self.outPutNode = 1
        self.wi = np.random.normal(0,1,self.numberOfFeatures*self.params['nh']).reshape(self.params['nh'],self.numberOfFeatures)
        self.wo = np.random.normal(0,1,self.params['nh'] *self.outPutNode).reshape(self.outPutNode,self.params['nh'])

        for i in range(self.params['epochs']):
            dataPointIndexList = np.arange(Xtrain.shape[0])
            stepSize = self.params['stepsize']/(i+1)
            np.random.shuffle(dataPointIndexList);
            for index in dataPointIndexList:
                self.updateWeights(Xtrain[index,:],ytrain[index],stepSize)


    def derivativeOfSigmoid(self,sigmoidValues):
        """ Gradient of standard sigmoid 1/(1+e^-x) """
        return sigmoidValues * (1 - sigmoidValues)


    def updateWeights(self,samplePoint, trueLable,stepSize):
        (ah,a0) = self._evaluate(samplePoint)
        delta2 = np.multiply(a0,1-a0)*(a0-trueLable)
        delta1 = delta2*self.derivativeOfSigmoid(ah)
        self.wo = np.subtract(self.wo, stepSize*delta2)
        self.wi = np.subtract(self.wi.T, stepSize*delta1)
        self.wi = self.wi.T

    def predict(self, Xtest):
        ytest = []
        for i in range(Xtest.shape[0]):
            (ah,a0) =self._evaluate(Xtest[i,:])
            if a0 >=0.5:
                ytest.append(1)
            else:
                ytest.append(0)
        return ytest


    # TODO: implement learn and predict functions                  

    
    def _evaluate(self, inputs):
        """ 
        Returns the output of the current neural network for the given input
        The underscore indicates that this is a private function to the class NeuralNet
        """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs))  

        # output activations
        ao = self.transfer(np.dot(self.wo,ah))
        
        return (ah, ao)





class LogitRegAlternative(Classifier):

    def __init__( self, parameters={} ):
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        
    # TODO: implement learn and predict functions                  


    def learn(self, Xtrain, ytrain):

        yt = np.copy(ytrain)
        numsamples = Xtrain.shape[0]

        initialWeights = np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T, Xtrain)), Xtrain.T), yt)
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T, Xtrain)), Xtrain.T), yt)

        eta = 0.3
        gradient = np.zeros(self.weights.shape[0])
        error = 100
        prevError=0
        currentPrediction = np.zeros(Xtrain.shape[0])

        while math.fabs(error - prevError) > 0.00001:

            prevPreduction = currentPrediction
            currentPrediction = self.predictProbablity(Xtrain)
            prevError = error

            diff = prevPreduction -currentPrediction
            error = np.linalg.norm(diff)
            print(error)
            if prevError < error:
                eta = eta/2
            print(self.weights)

            self.weights = self.weights + eta * np.dot(Xtrain.T, ((1 - 2 * ytrain) / np.sqrt(1 + np.square(currentPrediction)) + (currentPrediction / 1 + np.square(currentPrediction)))) / numsamples

    def sqrt(self,xw):
        return np.sqrt(1 + np.square(xw))

    def one_plus_xwSquare(self,xw):
        return (1 + np.square(xw))

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest = 0.5 * (1 + np.divide(ytest, np.sqrt(1 + np.square(ytest))))
        ytest[ytest<=0.5] = 0
        ytest[ytest > 0.5] = 1
        return ytest

    def predictProbablity(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return 0.5*(1+np.divide(ytest,np.sqrt(1 + np.square(ytest))))
