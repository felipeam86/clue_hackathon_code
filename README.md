 <p align="center"><img src="images/the_simptoms.png" width="400"></p>

This repository contains our contribution to the
[Clue-WATTx hackathon](http://cluehackathon.wattx.io/)

# 1. Usage
## 1.1 Dependencies
The code was developed with python 3.5 and the following libraries and
respective versions :
- pandas 0.19.2
- keras 2.0.1
- tensorflow-gpu 1.0.1 (It can also work with non gpu version)
- numpy 1.12.0
- joblib 0.10.3

## 1.2 Training
   First, you have to train the model using the train.py script.

   While training, the weights are automatically stored each time the
   validation loss decreases (in the /weights directory).
   The parameters by default train a simple
   LSTM model with  1 layer of 128 neurons over 15 epochs. It uses 100000 sequences
   for training and 50000 for testing. The input and output size is 16
   (i.e. 16 symptoms in input and 16 symptoms in output), 
   and sequences of 90 consecutive days are used for training and
   predicting. The parameters can be tweaked from the command line
   interface :


```bash
$ python train.py --help
usage: train.py [-h] [-N_train N_TRAIN] [-N_test N_TEST] [-N_epochs N_EPOCHS]
                [-batch_size BATCH_SIZE] [-input_size INPUT_SIZE]
                [-output_size OUTPUT_SIZE] [-maxlen MAXLEN]
                [-step_days STEP_DAYS] [-model {1,2}] [-weights WEIGHTS]
                [-debug]

optional arguments:
  -h, --help            show this help message and exit
  -N_train N_TRAIN      Number of sequences used for training. (default:
                        100000)
  -N_test N_TEST        Number of sequences used for testing. (default: 50000)
  -N_epochs N_EPOCHS    Number of epochs (default: 15)
  -batch_size BATCH_SIZE
                        Batch size (default: 256)
  -input_size INPUT_SIZE
                        Input size (default: 16)
  -output_size OUTPUT_SIZE
                        Output size (default: 16)
  -maxlen MAXLEN        Max length of the sequence (default: 90)
  -step_days STEP_DAYS  STEP_DAYS (default: 3)
  -model {1,2}          1 or 2 layers model (default: 1)
  -weights WEIGHTS      Where to store the weights after training (default:
                        None)
  -debug                If True, use a reduced subset of the data. (default:
                        False)


```

## 1.3 Prediction
   After model training, the predictions are made with predict.py.

   It automatically loads the pretrained weights assuming you use the
   exact same parameters as during training. The parameters can be tweaked from the
   command line interface :

```bash
usage: predict.py [-h] [-input_size INPUT_SIZE] [-output_size OUTPUT_SIZE]
                  [-maxlen MAXLEN] [-model {1,2}] [-weights WEIGHTS]

optional arguments:
  -h, --help            show this help message and exit
  -input_size INPUT_SIZE
                        Input size (default: 16)
  -output_size OUTPUT_SIZE
                        Output size (default: 16)
  -maxlen MAXLEN        Max length of the sequence (default: 90)
  -model {1,2}          1 or 2 layers model (default: 1)
  -weights WEIGHTS      Where to load pretrained weights (default: None)
```

## 1.4 Additional checks

   Before starting the training or prediction phase, ensure all csv files are added
   to the data/ directory. Due to privacy concerns these files have not been uploaded
   with the source code.

# 2. Preprocessing

The preprocessing steps are explained in detail on this notebook:
[1.0-Prepare-training-data](https://github.com/felipeam86/clue_hackathon_code/blob/master/notebooks/1.0-Prepare-training-data.ipynb)

# 3. Modeling
Our solution leverages neural networks, more specifically recurrent neural networks (RNN) with long short term memory (LSTM). RNNs are well suited to deal with time series, which is why we chose this approach.

## 3.1. Benefits of neural networks
Neural networks offer the benefit of being end-to-end solutions, i.e. if well architectured they deal with feature
engineering by themselves for the most part. For instance with a RNN there is no need to include whether the user was inactive on a specific day, or whether she was active but didn't experience a specific symptom. RNNs will determine this by themselves, if it is a useful feature for minimizing loss.

The other benefit is that neural networks often provide better performance than traditional machine learning techniques.
This has been observed for image recognition, image caption, speech recognition amongst others.

## 3.2 Drawbacks of neural networks
The main drawback of neural networks is the time it takes to train them. Convergence to a minima can be very time consuming,
in particular for RNNs which consist of many neural networks running in parallel, rapidly growing to millions of parameters to tune.
This has been a big challenge for this competition given the amount of data to process and the timeout set to 2 hours on the Statice platform.

Our solution was designed so that the RNN can be trained locally and weights are reused without further training on the Statice platform.
This speeds up processing, but this also means that training is performed on synthetic data that doesn't necessarily match well the real data.
As a result we observed big discrepancies between local performance and on the Statice platform.

Another drawback of NNs is the difficulty to interpret them. With millions of parameters and little human feature engineering,
understanding the logic of how the NN learns and predicts can be nearly impossible. With increasing concerns for transparency and new
EU regulations soon to be effective, the need to explain clearly the automated decisions may prevent the use of NNs in certain situations.
However, this is not an issue in this particular case, as the RNNs are not used to take clinical decisions, only as a suggestion engine of plausible
experienced symptoms throughout the cycle.

## 3.3 Architecture
We chose to explore two main architectures: 1 LSTM layer with 128 cells and 2 LSTM layers with 256 cells.

<p align="center"><img src="images/lstm.png" width="150"></p>
<p align="center">Illustration of a multi input/single output lstm</p>

Our RNNs are trained with a historical sequence of n days (by default 90) describing symptoms experienced by users (the input X), and the labels are
the symptoms experiences by the same users on the n+1 day (the output y)

Given the amount of users and length of the history, ideally we would like to train our network on all sequences of 90+1 days for every
woman, however this would generate too much data. Therefore we limit the number of training sequences to 100000 by default (but
this can be increased) and we chose to skip m (by default, STEP_DAYS = 3) step days of history for each woman. In other words, we look at the sequence of days
1 to 90, then 4 to 93, then 7 to 96 etc.

Other parameters include the number of symptoms to be predicted (by default 16 corresponding to the symptoms to be predicted in the hackathon, and we never modified that value), as
well as the number of symptoms to be used as input (by default the same 16 symptoms, but can be increase to
the full 81 symptoms). Our code also allows the user to include further data: day in cycle, whether user is experiencing her periods on a given day. We did not include these features for our tests due to training time constraints but we believe they may be useful.

Finally the number of epochs corresponds to the number of times the RNN will see the full set of training sequences. Typically we observe
that for a number of epochs the training and validation loss both decrease, then we reach a point where validation loss stagnates, and
finally the validation loss increases while the training loss keeps decreasing. This last phase corresponds to overfitting, and at that
point it is better to stop the training. We setup the network so the RNN weights are saved only when the validation loss improves, so
continuing training after reaching the overfit phase doesn't harm the model, but it is a pure waste of time.

# 4 Performance
On local machines the performance of RNN looked very promising. We used a PC with 16Gb RAM and a GPU GTX 960M to pre-train the models.
With the default parameters, the RNN took 21 minutes to train and we achieved a log loss on hold out set (validation)
 of 0.0531 after 15 epochs, as shown in the graph below.
 <p align="center"><img src="images/lstm_1_layers_16_input_size_16_output_size_90_maxlen.png" width="400"></p>
Using the same weights on the Statice platform the obtained log loss is 0.0761

Trained on the Statice platform with the same parameters, we obtain a log loss of 0.0748

The training phase can be tested in a more interactive manner using this notebook:
[3.0-Train-LSTM-visualize-log-loss](https://github.com/felipeam86/clue_hackathon_code/blob/master/notebooks/3.0-Train-LSTM-visualize-log-loss.ipynb).
The training/validation loss and accuracy evolution over epochs are displayed at the end of the training.

There may be several reasons why the performance on the remotely trained model is not as good as the performance on the synthetic data.
One highly likely reason is that the 100,000 sequences we used to train on the static platform are not representative enough of the full user base. The power of neural nets lies in utilising large datasets, therefore the parameter N_train should be increased to train on more samples (if hardware allows).

It is also expected that increasing the sequence length from 90 to 120 days and reducing the step to 1 day intead of 3 will lead
to better performances (if hardware allows).

# 5. Next steps
## 5.1 Add additional variables
Our solution didn't take into account several variables made available to us, in particular specifics about the user such as
age, weight, country etc. These information may be meaningful and could help improve performance.

Also, our intuition is that adding an additional variable "last day of cycle" would greatly help the RNN to improve prediction
on the first few days of the next cycle.

## 5.2 Improve the RNN architecture
There are two obvious areas where the RNN can be improved.

The first one is linked to the regularization technique. We used simple dropout of 50%, but it is know that for RNNs dropout
should only be applied to non recurrent layers, as described in this [paper](https://arxiv.org/pdf/1409.2329.pdf)

The second one is connected to statefulness of RNNs. Our RNN is stateless, however we are processing sequences which are related
to each other, therefore at training time we could use statefulness to improve network.

## 5.3 Test the solution on a remote platform equipped with GPU
Most of our attempts to train the RNN on the Statice platform failed due to the the 2 hours timeout, whereas they were
executing successfully locally on a PC equipped with GPU. Having a remote environment running a GPU would allow remote training
and would very likely lead to performances equivalent to those observed locally.

# 6. Lessons learned
This competition was the first hackathon that all members of the team ever attended. It has been a lot of fun, a lot of effort
and came with numerous teachings. Here are some of them

## 6.1 Neural networks
We loved working with RNNs. They are state of the art and the way forward for many applications. Despite the lack of results on the
Statice platform, the good results we obtained locally give us confidence that they are the right way to deal with the challenge
proposed by clue.

We will keep learning about them and experimenting with them in future assignments.

## 6.2 Statice platform
Working with the Statice platform was a good challenge. Given this was the first public test of the platform, there are lots of adjustments that can be made. Our main recommendations are the following:
- enable GPU instances for those using neural networds
- enable better tracking/logging of errors to avoid to many back and forth between the developer and the platform administrator
- setup multiple platforms with variable amount of data to enable quicker iterative process during the development phase that requires
intensive testing

Overall we were proud to be 'early adoptors' of the Statice platform, which will no doubt address important privacy concerns that might have held many companies from sharing their data in similar competitions.

## 6.3 Clue data
This readme document referred a lot to the technical approach and little to the data itself. This is in part because the approach we chose required minimal feature engineering of the data itself, although extensive preprocessing was necessary to train/predict using a RNN.

It is worth mentioning that being purely a team of males, all of us learnt a lot from the initial brief and the data itself. The breadth of symptoms experienced and reported by women surprised us and definitely give us a better understanding of women around us.

Thank you Clue for collecting this data and using it for the benefit of all!
