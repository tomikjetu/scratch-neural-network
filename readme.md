# Neural Network from scratch
A neural network from scratch using cross-validation using the [Iris Dataset](https://www.kaggle.com/datasets/himanshunakrani/iris-dataset)

Implementation based on - Fundamentals of Neural Networks- Laurene Fausett

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