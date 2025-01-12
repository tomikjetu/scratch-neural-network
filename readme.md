# Neural Network from scratch
> Implementation based on: Fundamentals of Neural Networks- Laurene Fausett

A neural network from scratch using cross-validation using the [Iris Dataset](https://www.kaggle.com/datasets/himanshunakrani/iris-dataset)


The model has 4 inputs, 4 hidden nodes and 3 output nodes. The hidden layer and the output layer use sigmoid activation function.


train.py
``` 
# Load data
data = pandas.read_csv('iris.csv')

# Neural network parameters
# Network of 4 inputs, 4 hidden, 3 outputs
weights = numpy.random.rand(4, 4)  # Input-to-hidden weights
biases = numpy.random.rand(4)  # Hidden layer biases

output_weights = numpy.random.rand(4, 3)  # Hidden-to-output weights
output_biases = numpy.random.rand(3)  # Output layer biases

learning_rate = 0.05
```

## Experiments
> Run test.py to reproduce with trained weights

After 473 epochs, the validation accuracy reached 100% with holdout accuracy of 100%
```
Epoch 469: Validation accuracy: 0.9629629629629629
Epoch 470: Validation accuracy: 0.8888888888888888
Epoch 471: Validation accuracy: 0.9259259259259259
Epoch 472: Validation accuracy: 0.9259259259259259
Epoch 473: Validation accuracy: 1.0
Training complete.
Holdout accuracy: 1.0
```
Accuraccy on the whole dataset was 93.33%

### Randomness
The cross validation is based on randomness, therefore it was lucky, that the first run of `train.py` achieved such "high" accuracy.  

After running the train process again for the second time, the model couldn't be trained in 1000 epochs, the `test.py` accuracy dropped to 70%,

The best attempt to try to reproduce the first results was at epoch 544 with holdout accuracy of 93% and whole dataset accuracy of 91%

This could be improved by using the K-fold cross-validation algorithm!!