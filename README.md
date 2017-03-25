 <p align="center"><img src="images/the_simptoms.png" width="400"></p>

# Usage

- pandas==0.19.2
- keras==2.0.1
- tensorflow-gpu==1.0.1
- numpy==1.12.0
- joblib==0.10.3

# Preprocessing

# Modeling
Our solution leverages neural networks, more specifically recurrent neural networkds (RNN) with long short term memory (LSTM).
RNNs are well suited to deal with time series, which is why we chose this approach.

## Benefits of neural networks
Neural networks offer the benefit of being end-to-end solutions, i.e. if well architectured they deal with the feature
engineering by themselves for the most part. For instance with RNN there is no need to bother whether the user was inactive on a
specific day, or whether she was active but didn't experience a symptom. RNNs will figure this by themselves.

The other benefit is that neural networks often provide better performance than tradional machine learning techniques.
This was observed for image recognition, image caption, speech recognition amongst others.

## Drawbacks of neural networks
The main drawback of neural networks is the time it takes to train them. Covergence to an minima can be very time consuming,
in particular for RNN which consist of many neural networks running in parallel, rapidely growing to millions of parameters to tune.
This has been a big challenge in this competition given the amount of data to process and the timeout set to 2 hours on the statice plateform.

Our solution was designed so that the NN can be trained locally and weights are reused without further training on the stative plateform.
This speeds up processing, but this also means that training is performed on synthetic data that doesn't necessarily match well the real data.
As a result we observed big discrepancies between performance observed locally and on the statice plateform.

Another drawback of NNs is the difficulty to interpret them. With millions of parameters and no or little human feature engineering,
understanding the logic of how the NN learns and predicts can be nearly impossible. In this particular case it may not be an issue,
but with increasing concerns for transparency and new EU regulations soon to be effective, the need to explain clearly the
algorithm decision tree may prevent the use of NNs

## Architecture
We chose to explore two main architectures: 1 with 128 neurons and 2 layers 256 neurons each RNNs.

Our RNNs are trained with a history of n days (by default 90) describing symptoms experienced by users (the X), and the labels are
the symptoms experiences by the same users on the n+1 day (the y)

Given the amount of users and length of history, ideally we would like to train our network on all sequences of 90+1 days for every
woman, however this would generate way too much data. Therefore we limit the number of training sequences to 100000 by default (but
this can be increased) and we chose to skip m (by default 3) days of history for which women. In other words, we look at sequence
1 to 90, then 4 to 93, then 7 to 96 etc.

Other parameters invovle the amount of symptoms to be predicted (by default 16, and we never modified that value), as
well as the nunmber of symptoms + other data to be used as input (by default the same 16 symptoms, but can be increase to
the full 81 symptoms, and more data such as day in cycle, whether user is experiencing her periods etc) can be taken into account.

Finally the number of epochs corresponds to the number of times the NN will see the full set of training sequences. Typically we observe
that for a number of epochs the training and validation loss both decrease, then we reach a point where validation loss stagnates, and
finally the validation loss increases while the training loss keeps decreasing. This last phase corresponds to overfitting, and at that
point it is better to stop the training. We setup the network so the NN weights are saved only when the validation loss improves, so
keeping training after reaching the overfit phase doesn't harm the model, but it is a pure waste of time.

## Performance
On local machines the performance of RNN looked very promising. We used a PC with 16Gb RAM and a GPU GTX 960M to pre-train the models.
With the default parameters, the RNN took 21 minutes to train and we achieved a log loss on hold out set (validation)
 of 0.053 after 15 epochs, as shown in the graph below.
 <p align="center"><img src="images/lstm_1_layers_16_input_size_16_output_size_90_maxlen.png" width="400"></p>
Using the same weights on the statice plateform the obtained log loss is XXX
Trained on the statice plateform with the same parameters, we obtain a log loss of XXX

# Next steps
## Add additional variables
Our solution didn't take into account several variables made available to us, in particular specifics about the user such as
age, weight, country etc. These information are most probably meaningful and could help improve performance

Also our intuition is that adding an additional variable "last day of cycle" would greatly help the RNN to improve prediction
on the first few days of the next cycle.

Improve architecture, in particular dropout as recommended in https://arxiv.org/pdf/1409.2329.pdf
Also stateful

## Test the solution on a remote platefrom equipped with GPU
Most of our attempt to train the RNN on the statice platefrom failed due to the the 2 hours timeout, whereas they were
executing successfuly locally on a PC equipped with GPU. Having a remote environment running a GPU would allow remote training
and would very likely lead to performances equivalent to those observed locally.


# Lessons learned
From the team

On NNs

On statice

On Clue