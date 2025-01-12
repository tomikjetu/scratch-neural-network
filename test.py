import numpy
import pandas

data = pandas.read_csv('IRIS.csv')

def load_weights():
    global weights, biases, output_weights, output_biases
    weights = numpy.load('weights/weights.npy')
    biases = numpy.load('weights/biases.npy')
    output_weights = numpy.load('weights/output_weights.npy')
    output_biases = numpy.load('weights/output_biases.npy')

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def forward(x): 
    hidden = sigmoid(numpy.dot(x, weights) + biases)
    return sigmoid(numpy.dot(hidden, output_weights) + output_biases)


if  __name__ == '__main__':
    def get_target(species):
        if species == 'setosa':
            return [1, 0, 0]
        elif species == 'versicolor':
            return [0, 1, 0]
        elif species == 'virginica':
            return [0, 0, 1]
        
    inputs = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values  
    targets = data[['species']].values 
    targets = numpy.array([get_target(species) for species in targets])
    
    load_weights()
    accuracy = 0
    for input, target in zip(inputs, targets):
        output = forward(input)
        if numpy.argmax(output) == numpy.argmax(target):
            accuracy += 1
    print('Accuracy: {:.2f}%'.format(accuracy / len(inputs) * 100))    
