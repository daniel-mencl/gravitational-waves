# Foundation model for gravitational waves

The goal of this work is to create a model, which could be used to inspect gravitational waves and learn more about them.
The ideal outcome is a model into which you could input the time series of a gravitational wave and receive all the information about not only the wave,
but more importantly what caused it, as this is computationally infeasible using traditional methods.
We also hope that such a model could be used to predict the continuation of a wave.

## Research

[Transformer-based Parameter Estimation in Statistics](https://arxiv.org/abs/2403.00019)
[Machine Learning Applications in Gravitational Wave Astronomy](https://inspirehep.net/literature/2747290)

## Methodology

The input wave will be in the form of a time series with a fixed time step.
Such a series is ideal for a transformer, in this work we will be using a decoder only transformer. 
We will not be using an embedding of any kind, just a positional encoding for the input.
The output will not be a probability of a token, but we will connect the last embedding of the output to a fully connected layer, which will output a number of values (usually just one) with no activation function.
The training will be done on generated data, as there isn't enough actual data for gravitational waves. This also allows us to generate new data every epoch to increase the generalization capability of the model.

## First steps

Before moving to (simulated) gravitational waves, we will test whether this works on less complicated signal. The signal will be a weighted combination of a sine wave and cosine squared wave.
The sine wave will be what we are interested in, while the cosine squared will be a kind of "noise". We will not worry about the amplitude (as we can normalize any wave) and the phase (while it will be included in the training set, so the model is shown waves at different starting points). The only value we care about is the frequency of the wave. Gravitational waves come in at around 100 Hz, so we will be trying waves between 85 and 115 Hz.

The training loop can be seen in [this notebook](training_example.ipynb), it uses functions from the src module.
Using this loop, we've trained two different models - one for inferring the frequency of the wave and another for predicting the next value.
Both of these can be found in the model folder.

### Frequency inference

The trained model and its' usage can be seen in [this notebook](signal_frequency_inference.ipynb).
In the notebook, you can how the model inputs look and how well it does at predicting the waves' frequencies.
The resulting MAE on the evaluation data (although since we are generating the data, this would change every time; it it also data the model hasn't seen, making it test data as well) is around 0.286.

### Next value prediction

The trained model and its' usage can be seen in [this notebook](signal_prediction.ipynb).
This way of training the model allows us to use it in a fashion similar to LLMs - take the input, predict the next value and repeat.
You can see how well the model is able to predict the continuation of a wave in the notebook.

## Continuation

Both of these models could be vastly improved by deepening the model, adding noise to the input (mainly for the second model) and generating more complicated data.
However, we can see that the techniques at least work on simpler inputs and we can try to train the models on simulated gravitational waves.
There is also hope that the models for parameter inference and next value prediction could be combined to obtain an actual foundation model.