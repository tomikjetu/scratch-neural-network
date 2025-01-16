# Neural Network from scratch
> Implementation based on: Fundamentals of Neural Networks- Laurene Fausett

A neural network from scratch using cross-validation using the [Iris Dataset](https://www.kaggle.com/datasets/himanshunakrani/iris-dataset)


The model has 4 inputs, 4 hidden nodes and 3 output nodes. The hidden layer and the output layer use sigmoid activation function. The consistency of training was increased after implementing the Nguyen Widrow initialization and K-Fold algorithm using the Sklearn library. 


train.py
``` 
# Load data
data = pandas.read_csv('iris.csv')

# Neural network parameters
# Network of 4 inputs, 4 hidden, 3 outputs
weights, biases = nguyen_widrow_init(4, 4)
output_weights, output_biases = nguyen_widrow_init(4, 3)

learning_rate = 0.05
```

## Experiments
> Run test.py to reproduce with trained weights

```
Fold 4, Epoch 146: Validation accuracy: 0.9655172413793103
Fold 4, Epoch 147: Validation accuracy: 0.9655172413793103
Fold 4, Epoch 148: Validation accuracy: 0.9655172413793103
Fold 4, Epoch 149: Validation accuracy: 0.9655172413793103
Fold 4 complete. Accuracy: 0.9655172413793103
Fold 5, Epoch 0: Validation accuracy: 1.0
Fold 5 complete. Accuracy: 1.0
K-Fold Cross-Validation Complete. Mean Accuracy: 0.9724135376756067
Fold 4 complete. Accuracy: 0.9655172413793103
Fold 5, Epoch 0: Validation accuracy: 1.0
Fold 5 complete. Accuracy: 1.0
K-Fold Cross-Validation Complete. Mean Accuracy: 0.9724135376756067
K-Fold Cross-Validation Complete. Mean Accuracy: 0.9724135376756067
Training complete.
Holdout accuracy: 1.0
> python test.py
Accuracy: 96.00%
```
