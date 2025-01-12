import numpy
import pandas

# Load data
data = pandas.read_csv('iris.csv')

# Neural network parameters
# Network of 4 inputs, 4 hidden, 3 outputs
weights = numpy.random.rand(4, 4)  # Input-to-hidden weights
biases = numpy.random.rand(4)  # Hidden layer biases

output_weights = numpy.random.rand(4, 3)  # Hidden-to-output weights
output_biases = numpy.random.rand(3)  # Output layer biases

learning_rate = 0.05

# Activation functions
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoid_derivative(x):
    # return sigmoid(x) * (1 - sigmoid(x))
    return x * (1 - x)

# Training function
def train(inputs, target):
    global weights, biases, output_weights, output_biases

    inputs = numpy.atleast_2d(inputs)
    target = numpy.atleast_2d(target)

    # Forward pass
    hidden = sigmoid(numpy.dot(inputs, weights) + biases)  # Hidden layer activations
    output = sigmoid(numpy.dot(hidden, output_weights) + output_biases)  # Output layer activations

    # Compute output error and deltas
    error_output = (target - output) * sigmoid_derivative(output)
    delta_output_weights = learning_rate * numpy.dot(hidden.T, error_output)    
    delta_output_biases = learning_rate * numpy.sum(error_output, axis=0)

    # Compute hidden layer error and deltas
    error_hidden = numpy.dot(error_output, output_weights.T) * sigmoid_derivative(hidden)
    delta_weights = learning_rate * numpy.dot(inputs.T, error_hidden)
    delta_biases = learning_rate * numpy.sum(error_hidden, axis=0)

    # Update weights and biases
    weights += delta_weights
    biases += delta_biases
    output_weights += delta_output_weights
    output_biases += delta_output_biases

def forward(x): 
    hidden = sigmoid(numpy.dot(x, weights) + biases)
    return sigmoid(numpy.dot(hidden, output_weights) + output_biases)

def save_weights():
    numpy.save('weights/weights.npy', weights)
    numpy.save('weights/biases.npy', biases)
    numpy.save('weights/output_weights.npy', output_weights)
    numpy.save('weights/output_biases.npy', output_biases)

def load_weights():
    global weights, biases, output_weights, output_biases
    weights = numpy.load('weights/weights.npy')
    biases = numpy.load('weights/biases.npy')
    output_weights = numpy.load('weights/output_weights.npy')
    output_biases = numpy.load('weights/output_biases.npy')

def split_data(x, test_size=0.2):
    from random import sample

    inputs, targets = x
    if len(inputs) != len(targets):
        raise ValueError("The lengths of inputs and targets must be the same.")
   
    data_length = len(inputs)
    test_count = int(data_length * test_size)
    
    # Randomly select indices for the test set
    test_indices = sample(range(data_length), test_count)
    
    # Create test and training sets based on selected indices
    test_inputs = [inputs[i] for i in test_indices]
    test_targets = [targets[i] for i in test_indices]
    
    train_inputs = [inputs[i] for i in range(data_length) if i not in test_indices]
    train_targets = [targets[i] for i in range(data_length) if i not in test_indices]
    
    return [train_inputs, train_targets], [test_inputs, test_targets]



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
    data = [inputs, targets]
    train_data, holdout_data = split_data(data, test_size=0.1)


    # Training with validation
    for epoch in range(1000):  
        epoch_train, epoch_valid = split_data(train_data, test_size=0.2)
        for i in range(len(epoch_train[0])):
            train(epoch_train[0][i], epoch_train[1][i])

        validation_accuracy = 0
        for i in range(len(epoch_valid[0])):
            prediction = forward(epoch_valid[0][i])
            if numpy.argmax(prediction) == numpy.argmax(epoch_valid[1][i]):
                validation_accuracy += 1
        validation_accuracy /= len(epoch_valid[0])
        print(f"Epoch {epoch}: Validation accuracy: {validation_accuracy}")
        if validation_accuracy == 1:
            break

    save_weights()
    print("Training complete.")
    
    holdout_accuracy = 0
    for i in range(len(holdout_data[0])):
        prediction = forward(holdout_data[0][i])
        if numpy.argmax(prediction) == numpy.argmax(holdout_data[1][i]):
            holdout_accuracy += 1
    holdout_accuracy /= len(holdout_data[0])
    print(f"Holdout accuracy: {holdout_accuracy}")