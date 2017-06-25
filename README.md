
# Predicting NBA players positions using Keras #

In this notebook we will build a neural net to predict the positions of NBA players using the [Keras](https://keras.io) library.


```python
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
```

## Data preparation ##

We will use the Kaggle dataset ["NBA Players stats since 1950"](https://www.kaggle.com/drgilermo/nba-players-stats), with stats for all players since 1950. We will take special interest in how the pass of time affects to the position of each player, and the definition of the positions themselves (a Small Forward, for example, was absolutely different in the 60's than now)


```python
stats = pd.read_csv(r'data/Seasons_Stats.csv', index_col=0)
```

The file ```Seasons_Stats.csv``` contains the statics of all players since 1950. First, we drop a couple of blank columns, and the "Tm" column, that contains the team.


```python
stats = pd.read_csv(r'data/Seasons_Stats.csv', index_col=0)
stats_clean = stats.drop(['blanl', 'blank2', 'Tm'], axis=1)
```


```python
stats_clean.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Player</th>
      <th>Pos</th>
      <th>Age</th>
      <th>G</th>
      <th>GS</th>
      <th>MP</th>
      <th>PER</th>
      <th>TS%</th>
      <th>3PAr</th>
      <th>...</th>
      <th>FT%</th>
      <th>ORB</th>
      <th>DRB</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>TOV</th>
      <th>PF</th>
      <th>PTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1950.0</td>
      <td>Curly Armstrong</td>
      <td>G-F</td>
      <td>31.0</td>
      <td>63.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.368</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.705</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>176.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>217.0</td>
      <td>458.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1950.0</td>
      <td>Cliff Barker</td>
      <td>SG</td>
      <td>29.0</td>
      <td>49.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.435</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.708</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>109.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>99.0</td>
      <td>279.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1950.0</td>
      <td>Leo Barnhorst</td>
      <td>SF</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.394</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.698</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>140.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>192.0</td>
      <td>438.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1950.0</td>
      <td>Ed Bartels</td>
      <td>F</td>
      <td>24.0</td>
      <td>15.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.312</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.559</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.0</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1950.0</td>
      <td>Ed Bartels</td>
      <td>F</td>
      <td>24.0</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.308</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.548</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.0</td>
      <td>59.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49 columns</p>
</div>



A second file, ```players.csv```, contains static information for each player, as height, weight, etc.


```python
players = pd.read_csv(r'data/players.csv', index_col=0)
players.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>height</th>
      <th>weight</th>
      <th>collage</th>
      <th>born</th>
      <th>birth_city</th>
      <th>birth_state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Curly Armstrong</td>
      <td>180.0</td>
      <td>77.0</td>
      <td>Indiana University</td>
      <td>1918.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cliff Barker</td>
      <td>188.0</td>
      <td>83.0</td>
      <td>University of Kentucky</td>
      <td>1921.0</td>
      <td>Yorktown</td>
      <td>Indiana</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Leo Barnhorst</td>
      <td>193.0</td>
      <td>86.0</td>
      <td>University of Notre Dame</td>
      <td>1924.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ed Bartels</td>
      <td>196.0</td>
      <td>88.0</td>
      <td>North Carolina State University</td>
      <td>1925.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ralph Beard</td>
      <td>178.0</td>
      <td>79.0</td>
      <td>University of Kentucky</td>
      <td>1927.0</td>
      <td>Hardinsburg</td>
      <td>Kentucky</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gene Berce</td>
      <td>180.0</td>
      <td>79.0</td>
      <td>Marquette University</td>
      <td>1926.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Charlie Black</td>
      <td>196.0</td>
      <td>90.0</td>
      <td>University of Kansas</td>
      <td>1921.0</td>
      <td>Arco</td>
      <td>Idaho</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Nelson Bobb</td>
      <td>183.0</td>
      <td>77.0</td>
      <td>Temple University</td>
      <td>1924.0</td>
      <td>Philadelphia</td>
      <td>Pennsylvania</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Jake Bornheimer</td>
      <td>196.0</td>
      <td>90.0</td>
      <td>Muhlenberg College</td>
      <td>1927.0</td>
      <td>New Brunswick</td>
      <td>New Jersey</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Vince Boryla</td>
      <td>196.0</td>
      <td>95.0</td>
      <td>University of Denver</td>
      <td>1927.0</td>
      <td>East Chicago</td>
      <td>Indiana</td>
    </tr>
  </tbody>
</table>
</div>



We merge both tables, and do some data cleaning:

* Keep only players with more than 400 minutes for each season (with a 82 games regular season, thats around 5 minutes per game. Players with less than that will be only anecdotical, and will distort the analysis).
* Replace the \* sign in some of the names
* For the stats that represent total values (others, as TS%, represent percentages), we will take the values per 36 minutes. The reason is to judge every player according to his characteristics, not the time he was on the floor.


```python
data = pd.merge(stats_clean, players[['Player', 'height', 'weight']], left_on='Player', right_on='Player', right_index=False,
      how='left', sort=False).fillna(value=0)
data = data[~(data['Pos']==0) & (data['MP'] > 200)]
data.reset_index(inplace=True, drop=True)
data['Player'] = data['Player'].str.replace('*','')

totals = ['PER', 'OWS', 'DWS', 'WS', 'OBPM', 'DBPM', 'BPM', 'VORP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA',
         'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

for col in totals:
    data[col] = 36 * data[col] / data['MP']
```


```python
data.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Player</th>
      <th>Pos</th>
      <th>Age</th>
      <th>G</th>
      <th>GS</th>
      <th>MP</th>
      <th>PER</th>
      <th>TS%</th>
      <th>3PAr</th>
      <th>...</th>
      <th>DRB</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>TOV</th>
      <th>PF</th>
      <th>PTS</th>
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19695</th>
      <td>2017.0</td>
      <td>Thaddeus Young</td>
      <td>PF</td>
      <td>28.0</td>
      <td>74.0</td>
      <td>74.0</td>
      <td>2237.0</td>
      <td>0.239785</td>
      <td>0.562</td>
      <td>0.172</td>
      <td>...</td>
      <td>5.117568</td>
      <td>7.225749</td>
      <td>1.963344</td>
      <td>1.834600</td>
      <td>0.482789</td>
      <td>1.544926</td>
      <td>2.172553</td>
      <td>13.099687</td>
      <td>203.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>19696</th>
      <td>2017.0</td>
      <td>Cody Zeller</td>
      <td>PF</td>
      <td>24.0</td>
      <td>62.0</td>
      <td>58.0</td>
      <td>1725.0</td>
      <td>0.348522</td>
      <td>0.604</td>
      <td>0.002</td>
      <td>...</td>
      <td>5.634783</td>
      <td>8.452174</td>
      <td>2.066087</td>
      <td>1.293913</td>
      <td>1.210435</td>
      <td>1.356522</td>
      <td>3.944348</td>
      <td>13.335652</td>
      <td>213.0</td>
      <td>108.0</td>
    </tr>
    <tr>
      <th>19697</th>
      <td>2017.0</td>
      <td>Tyler Zeller</td>
      <td>C</td>
      <td>27.0</td>
      <td>51.0</td>
      <td>5.0</td>
      <td>525.0</td>
      <td>0.891429</td>
      <td>0.508</td>
      <td>0.006</td>
      <td>...</td>
      <td>5.554286</td>
      <td>8.502857</td>
      <td>2.880000</td>
      <td>0.480000</td>
      <td>1.440000</td>
      <td>1.371429</td>
      <td>4.182857</td>
      <td>12.205714</td>
      <td>213.0</td>
      <td>114.0</td>
    </tr>
    <tr>
      <th>19698</th>
      <td>2017.0</td>
      <td>Paul Zipser</td>
      <td>SF</td>
      <td>22.0</td>
      <td>44.0</td>
      <td>18.0</td>
      <td>843.0</td>
      <td>0.294662</td>
      <td>0.503</td>
      <td>0.448</td>
      <td>...</td>
      <td>4.697509</td>
      <td>5.338078</td>
      <td>1.537367</td>
      <td>0.640569</td>
      <td>0.683274</td>
      <td>1.708185</td>
      <td>3.330961</td>
      <td>10.249110</td>
      <td>203.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>19699</th>
      <td>2017.0</td>
      <td>Ivica Zubac</td>
      <td>C</td>
      <td>19.0</td>
      <td>38.0</td>
      <td>11.0</td>
      <td>609.0</td>
      <td>1.004926</td>
      <td>0.547</td>
      <td>0.013</td>
      <td>...</td>
      <td>6.975369</td>
      <td>9.399015</td>
      <td>1.773399</td>
      <td>0.827586</td>
      <td>1.950739</td>
      <td>1.773399</td>
      <td>3.901478</td>
      <td>16.788177</td>
      <td>216.0</td>
      <td>120.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 51 columns</p>
</div>



We will train a neural network with this data, to try to predict the position of each player.

A way we didn't follow was to transform the positions into numbers from 1 to 5 (1 for a PG, 2 for a SG, 1.5 for a PG-SG, and so on, until 5 for a C), and use the network for regression instead of classification. But we wanted to see if the network was able to predict labels as "SG-PF", so we decided to work with the categorical labels. Another reason is that this makes this study more easily portable to other areas.

We convert our DataFrame into a matrix X with the inputs, and a vector y with the labels. We scale the inputs and encode the outputs into dummy variables using the corresponding ```sklearn``` utilities.

Instead of a stochastic partition, we decided to use the 2017 season as our test data, and all the previous as the train set.


```python
X = data.drop(['Player', 'Pos', 'G', 'GS', 'MP'], axis=1).as_matrix()
y = data['Pos'].as_matrix()

encoder = LabelBinarizer()
y_cat = encoder.fit_transform(y)
nlabels = len(encoder.classes_)

scaler =StandardScaler()
Xnorm = scaler.fit_transform(X)

stats2017 = (data['Year'] == 2017)
X_train = Xnorm[~stats2017]
y_train = y_cat[~stats2017]
X_test = Xnorm[stats2017]
y_test = y_cat[stats2017]
```

## Neural network training ##

We build using Keras (with Tensorflow as beckend) a neural network with two hidden layers. We will use relu activations, except for the last one, where we use a softmax to properly obtain the label probability. We will use a 20% of the data as a validation set, to make sure we are not overfitting.


```python
model = Sequential()
model.add(Dense(40, activation='relu', input_dim=46))
model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nlabels, activation='softmax'))
```


```python
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```


```python
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0.2, verbose=1)
```

    Train on 15378 samples, validate on 3845 samples
    Epoch 1/200
    15378/15378 [==============================] - 1s - loss: 2.8803 - acc: 0.1578 - val_loss: 1.9745 - val_acc: 0.4107
    Epoch 2/200
    15378/15378 [==============================] - 0s - loss: 2.0452 - acc: 0.3210 - val_loss: 1.4504 - val_acc: 0.4661
    Epoch 3/200
    15378/15378 [==============================] - 0s - loss: 1.6946 - acc: 0.3707 - val_loss: 1.2529 - val_acc: 0.5394
    Epoch 4/200
    15378/15378 [==============================] - 0s - loss: 1.5110 - acc: 0.4004 - val_loss: 1.1616 - val_acc: 0.5664
    Epoch 5/200
    15378/15378 [==============================] - 0s - loss: 1.3971 - acc: 0.4282 - val_loss: 1.1100 - val_acc: 0.6031
    Epoch 6/200
    15378/15378 [==============================] - 0s - loss: 1.3392 - acc: 0.4400 - val_loss: 1.0867 - val_acc: 0.6343
    Epoch 7/200
    15378/15378 [==============================] - 0s - loss: 1.2855 - acc: 0.4607 - val_loss: 1.0710 - val_acc: 0.6502
    Epoch 8/200
    15378/15378 [==============================] - 0s - loss: 1.2508 - acc: 0.4808 - val_loss: 1.0556 - val_acc: 0.6364
    Epoch 9/200
    15378/15378 [==============================] - 0s - loss: 1.2192 - acc: 0.4835 - val_loss: 1.0575 - val_acc: 0.6518
    Epoch 10/200
    15378/15378 [==============================] - 0s - loss: 1.1847 - acc: 0.5014 - val_loss: 1.0365 - val_acc: 0.6689
    Epoch 11/200
    15378/15378 [==============================] - 0s - loss: 1.1752 - acc: 0.5105 - val_loss: 1.0345 - val_acc: 0.6627
    Epoch 12/200
    15378/15378 [==============================] - 0s - loss: 1.1432 - acc: 0.5209 - val_loss: 1.0213 - val_acc: 0.6635
    Epoch 13/200
    15378/15378 [==============================] - 0s - loss: 1.1337 - acc: 0.5254 - val_loss: 1.0245 - val_acc: 0.6668
    Epoch 14/200
    15378/15378 [==============================] - 0s - loss: 1.1146 - acc: 0.5313 - val_loss: 1.0121 - val_acc: 0.6715
    Epoch 15/200
    15378/15378 [==============================] - 0s - loss: 1.1045 - acc: 0.5416 - val_loss: 1.0194 - val_acc: 0.6544
    Epoch 16/200
    15378/15378 [==============================] - 0s - loss: 1.0943 - acc: 0.5451 - val_loss: 1.0252 - val_acc: 0.6533
    Epoch 17/200
    15378/15378 [==============================] - 0s - loss: 1.0931 - acc: 0.5430 - val_loss: 1.0063 - val_acc: 0.6679
    Epoch 18/200
    15378/15378 [==============================] - 0s - loss: 1.0771 - acc: 0.5486 - val_loss: 1.0010 - val_acc: 0.6694
    Epoch 19/200
    15378/15378 [==============================] - 0s - loss: 1.0685 - acc: 0.5513 - val_loss: 1.0011 - val_acc: 0.6624
    Epoch 20/200
    15378/15378 [==============================] - 0s - loss: 1.0587 - acc: 0.5583 - val_loss: 0.9910 - val_acc: 0.6666
    Epoch 21/200
    15378/15378 [==============================] - 0s - loss: 1.0556 - acc: 0.5655 - val_loss: 0.9871 - val_acc: 0.6767
    Epoch 22/200
    15378/15378 [==============================] - 0s - loss: 1.0433 - acc: 0.5668 - val_loss: 0.9917 - val_acc: 0.6715
    Epoch 23/200
    15378/15378 [==============================] - 0s - loss: 1.0419 - acc: 0.5689 - val_loss: 0.9779 - val_acc: 0.6705
    Epoch 24/200
    15378/15378 [==============================] - 0s - loss: 1.0312 - acc: 0.5742 - val_loss: 0.9855 - val_acc: 0.6754
    Epoch 25/200
    15378/15378 [==============================] - 0s - loss: 1.0329 - acc: 0.5699 - val_loss: 0.9770 - val_acc: 0.6765
    Epoch 26/200
    15378/15378 [==============================] - 0s - loss: 1.0379 - acc: 0.5678 - val_loss: 0.9869 - val_acc: 0.6463
    Epoch 27/200
    15378/15378 [==============================] - 0s - loss: 1.0217 - acc: 0.5832 - val_loss: 0.9847 - val_acc: 0.6606
    Epoch 28/200
    15378/15378 [==============================] - 0s - loss: 1.0230 - acc: 0.5789 - val_loss: 0.9795 - val_acc: 0.6801
    Epoch 29/200
    15378/15378 [==============================] - 0s - loss: 1.0237 - acc: 0.5779 - val_loss: 0.9777 - val_acc: 0.6791
    Epoch 30/200
    15378/15378 [==============================] - 0s - loss: 1.0117 - acc: 0.5856 - val_loss: 0.9694 - val_acc: 0.6762
    Epoch 31/200
    15378/15378 [==============================] - 0s - loss: 1.0108 - acc: 0.5798 - val_loss: 0.9634 - val_acc: 0.6720
    Epoch 32/200
    15378/15378 [==============================] - 0s - loss: 1.0079 - acc: 0.5931 - val_loss: 0.9608 - val_acc: 0.6687
    Epoch 33/200
    15378/15378 [==============================] - 0s - loss: 1.0005 - acc: 0.5910 - val_loss: 0.9637 - val_acc: 0.6666
    Epoch 34/200
    15378/15378 [==============================] - 0s - loss: 0.9990 - acc: 0.5951 - val_loss: 0.9625 - val_acc: 0.6765
    Epoch 35/200
    15378/15378 [==============================] - 0s - loss: 0.9981 - acc: 0.5901 - val_loss: 0.9537 - val_acc: 0.6710
    Epoch 36/200
    15378/15378 [==============================] - 0s - loss: 0.9973 - acc: 0.5897 - val_loss: 0.9637 - val_acc: 0.6746
    Epoch 37/200
    15378/15378 [==============================] - 0s - loss: 0.9909 - acc: 0.5970 - val_loss: 0.9533 - val_acc: 0.6726
    Epoch 38/200
    15378/15378 [==============================] - 0s - loss: 0.9864 - acc: 0.5976 - val_loss: 0.9606 - val_acc: 0.6817
    Epoch 39/200
    15378/15378 [==============================] - 0s - loss: 0.9822 - acc: 0.6025 - val_loss: 0.9549 - val_acc: 0.6819
    Epoch 40/200
    15378/15378 [==============================] - 0s - loss: 0.9864 - acc: 0.5979 - val_loss: 0.9475 - val_acc: 0.6744
    Epoch 41/200
    15378/15378 [==============================] - 0s - loss: 0.9798 - acc: 0.6010 - val_loss: 0.9476 - val_acc: 0.6687
    Epoch 42/200
    15378/15378 [==============================] - 0s - loss: 0.9764 - acc: 0.5987 - val_loss: 0.9479 - val_acc: 0.6650
    Epoch 43/200
    15378/15378 [==============================] - 0s - loss: 0.9791 - acc: 0.6012 - val_loss: 0.9558 - val_acc: 0.6830
    Epoch 44/200
    15378/15378 [==============================] - 0s - loss: 0.9780 - acc: 0.6004 - val_loss: 0.9455 - val_acc: 0.6757
    Epoch 45/200
    15378/15378 [==============================] - 0s - loss: 0.9745 - acc: 0.6001 - val_loss: 0.9574 - val_acc: 0.6754
    Epoch 46/200
    15378/15378 [==============================] - 0s - loss: 0.9722 - acc: 0.6020 - val_loss: 0.9449 - val_acc: 0.6804
    Epoch 47/200
    15378/15378 [==============================] - 0s - loss: 0.9669 - acc: 0.6087 - val_loss: 0.9541 - val_acc: 0.6590
    Epoch 48/200
    15378/15378 [==============================] - 0s - loss: 0.9647 - acc: 0.6072 - val_loss: 0.9358 - val_acc: 0.6720
    Epoch 49/200
    15378/15378 [==============================] - 0s - loss: 0.9617 - acc: 0.6056 - val_loss: 0.9390 - val_acc: 0.6702
    Epoch 50/200
    15378/15378 [==============================] - 0s - loss: 0.9627 - acc: 0.6043 - val_loss: 0.9378 - val_acc: 0.6861
    Epoch 51/200
    15378/15378 [==============================] - 0s - loss: 0.9610 - acc: 0.6102 - val_loss: 0.9418 - val_acc: 0.6624
    Epoch 52/200
    15378/15378 [==============================] - 0s - loss: 0.9532 - acc: 0.6139 - val_loss: 0.9383 - val_acc: 0.6775
    Epoch 53/200
    15378/15378 [==============================] - 0s - loss: 0.9469 - acc: 0.6113 - val_loss: 0.9354 - val_acc: 0.6817
    Epoch 54/200
    15378/15378 [==============================] - 0s - loss: 0.9455 - acc: 0.6134 - val_loss: 0.9344 - val_acc: 0.6707
    Epoch 55/200
    15378/15378 [==============================] - 0s - loss: 0.9450 - acc: 0.6093 - val_loss: 0.9384 - val_acc: 0.6913
    Epoch 56/200
    15378/15378 [==============================] - 0s - loss: 0.9463 - acc: 0.6169 - val_loss: 0.9437 - val_acc: 0.6814
    Epoch 57/200
    15378/15378 [==============================] - 0s - loss: 0.9502 - acc: 0.6059 - val_loss: 0.9331 - val_acc: 0.6887
    Epoch 58/200
    15378/15378 [==============================] - 0s - loss: 0.9431 - acc: 0.6167 - val_loss: 0.9288 - val_acc: 0.6843
    Epoch 59/200
    15378/15378 [==============================] - 0s - loss: 0.9338 - acc: 0.6182 - val_loss: 0.9364 - val_acc: 0.6895
    Epoch 60/200
    15378/15378 [==============================] - 0s - loss: 0.9538 - acc: 0.6164 - val_loss: 0.9268 - val_acc: 0.6835
    Epoch 61/200
    15378/15378 [==============================] - 0s - loss: 0.9384 - acc: 0.6141 - val_loss: 0.9444 - val_acc: 0.6850
    Epoch 62/200
    15378/15378 [==============================] - 0s - loss: 0.9386 - acc: 0.6270 - val_loss: 0.9290 - val_acc: 0.6762
    Epoch 63/200
    15378/15378 [==============================] - 0s - loss: 0.9388 - acc: 0.6232 - val_loss: 0.9283 - val_acc: 0.6780
    Epoch 64/200
    15378/15378 [==============================] - 0s - loss: 0.9378 - acc: 0.6192 - val_loss: 0.9248 - val_acc: 0.6663
    Epoch 65/200
    15378/15378 [==============================] - 0s - loss: 0.9353 - acc: 0.6245 - val_loss: 0.9239 - val_acc: 0.6770
    Epoch 66/200
    15378/15378 [==============================] - 0s - loss: 0.9244 - acc: 0.6318 - val_loss: 0.9230 - val_acc: 0.6700
    Epoch 67/200
    15378/15378 [==============================] - 0s - loss: 0.9370 - acc: 0.6211 - val_loss: 0.9182 - val_acc: 0.6796
    Epoch 68/200
    15378/15378 [==============================] - 0s - loss: 0.9343 - acc: 0.6192 - val_loss: 0.9230 - val_acc: 0.6900
    Epoch 69/200
    15378/15378 [==============================] - 0s - loss: 0.9301 - acc: 0.6221 - val_loss: 0.9368 - val_acc: 0.6687
    Epoch 70/200
    15378/15378 [==============================] - 0s - loss: 0.9159 - acc: 0.6281 - val_loss: 0.9221 - val_acc: 0.6697
    Epoch 71/200
    15378/15378 [==============================] - 0s - loss: 0.9229 - acc: 0.6288 - val_loss: 0.9239 - val_acc: 0.6923
    Epoch 72/200
    15378/15378 [==============================] - 0s - loss: 0.9236 - acc: 0.6215 - val_loss: 0.9175 - val_acc: 0.6726
    Epoch 73/200
    15378/15378 [==============================] - 0s - loss: 0.9296 - acc: 0.6254 - val_loss: 0.9120 - val_acc: 0.6746
    Epoch 74/200
    15378/15378 [==============================] - 0s - loss: 0.9258 - acc: 0.6250 - val_loss: 0.9172 - val_acc: 0.6936
    Epoch 75/200
    15378/15378 [==============================] - 0s - loss: 0.9193 - acc: 0.6271 - val_loss: 0.9177 - val_acc: 0.6752
    Epoch 76/200
    15378/15378 [==============================] - 0s - loss: 0.9168 - acc: 0.6335 - val_loss: 0.9113 - val_acc: 0.6765
    Epoch 77/200
    15378/15378 [==============================] - 0s - loss: 0.9163 - acc: 0.6311 - val_loss: 0.9156 - val_acc: 0.6908
    Epoch 78/200
    15378/15378 [==============================] - 0s - loss: 0.9194 - acc: 0.6283 - val_loss: 0.9111 - val_acc: 0.6879
    Epoch 79/200
    15378/15378 [==============================] - 0s - loss: 0.9185 - acc: 0.6297 - val_loss: 0.9101 - val_acc: 0.6908
    Epoch 80/200
    15378/15378 [==============================] - 0s - loss: 0.9185 - acc: 0.6245 - val_loss: 0.9072 - val_acc: 0.6814
    Epoch 81/200
    15378/15378 [==============================] - 0s - loss: 0.9128 - acc: 0.6332 - val_loss: 0.9073 - val_acc: 0.6804
    Epoch 82/200
    15378/15378 [==============================] - 0s - loss: 0.9096 - acc: 0.6355 - val_loss: 0.9034 - val_acc: 0.6822
    Epoch 83/200
    15378/15378 [==============================] - 0s - loss: 0.9124 - acc: 0.6314 - val_loss: 0.9032 - val_acc: 0.6923
    Epoch 84/200
    15378/15378 [==============================] - 0s - loss: 0.9180 - acc: 0.6275 - val_loss: 0.9061 - val_acc: 0.6970
    Epoch 85/200
    15378/15378 [==============================] - 0s - loss: 0.9113 - acc: 0.6314 - val_loss: 0.9032 - val_acc: 0.6931
    Epoch 86/200
    15378/15378 [==============================] - 0s - loss: 0.9046 - acc: 0.6362 - val_loss: 0.9006 - val_acc: 0.6843
    Epoch 87/200
    15378/15378 [==============================] - 0s - loss: 0.9088 - acc: 0.6307 - val_loss: 0.9023 - val_acc: 0.6923
    Epoch 88/200
    15378/15378 [==============================] - 0s - loss: 0.9003 - acc: 0.6353 - val_loss: 0.9001 - val_acc: 0.6947
    Epoch 89/200
    15378/15378 [==============================] - 0s - loss: 0.8992 - acc: 0.6371 - val_loss: 0.9020 - val_acc: 0.6947
    Epoch 90/200
    15378/15378 [==============================] - 0s - loss: 0.9130 - acc: 0.6347 - val_loss: 0.8998 - val_acc: 0.6710
    Epoch 91/200
    15378/15378 [==============================] - 0s - loss: 0.9102 - acc: 0.6364 - val_loss: 0.9070 - val_acc: 0.6923
    Epoch 92/200
    15378/15378 [==============================] - 0s - loss: 0.9076 - acc: 0.6366 - val_loss: 0.8971 - val_acc: 0.6892
    Epoch 93/200
    15378/15378 [==============================] - 0s - loss: 0.9066 - acc: 0.6329 - val_loss: 0.9104 - val_acc: 0.6874
    Epoch 94/200
    15378/15378 [==============================] - 0s - loss: 0.8994 - acc: 0.6405 - val_loss: 0.9086 - val_acc: 0.6809
    Epoch 95/200
    15378/15378 [==============================] - 0s - loss: 0.8897 - acc: 0.6380 - val_loss: 0.8921 - val_acc: 0.6871
    Epoch 96/200
    15378/15378 [==============================] - 0s - loss: 0.8914 - acc: 0.6438 - val_loss: 0.8923 - val_acc: 0.6754
    Epoch 97/200
    15378/15378 [==============================] - 0s - loss: 0.8952 - acc: 0.6472 - val_loss: 0.8935 - val_acc: 0.6757
    Epoch 98/200
    15378/15378 [==============================] - 0s - loss: 0.8945 - acc: 0.6404 - val_loss: 0.8927 - val_acc: 0.6897
    Epoch 99/200
    15378/15378 [==============================] - 0s - loss: 0.9008 - acc: 0.6434 - val_loss: 0.8978 - val_acc: 0.6897
    Epoch 100/200
    15378/15378 [==============================] - 0s - loss: 0.8926 - acc: 0.6369 - val_loss: 0.8980 - val_acc: 0.6921
    Epoch 101/200
    15378/15378 [==============================] - 0s - loss: 0.8965 - acc: 0.6427 - val_loss: 0.8875 - val_acc: 0.6861
    Epoch 102/200
    15378/15378 [==============================] - 0s - loss: 0.8896 - acc: 0.6442 - val_loss: 0.8865 - val_acc: 0.6967
    Epoch 103/200
    15378/15378 [==============================] - 0s - loss: 0.8893 - acc: 0.6457 - val_loss: 0.8854 - val_acc: 0.6897
    Epoch 104/200
    15378/15378 [==============================] - 0s - loss: 0.8837 - acc: 0.6459 - val_loss: 0.8850 - val_acc: 0.6954
    Epoch 105/200
    15378/15378 [==============================] - 0s - loss: 0.8882 - acc: 0.6449 - val_loss: 0.8952 - val_acc: 0.6908
    Epoch 106/200
    15378/15378 [==============================] - 0s - loss: 0.8831 - acc: 0.6496 - val_loss: 0.8822 - val_acc: 0.6874
    Epoch 107/200
    15378/15378 [==============================] - 0s - loss: 0.8893 - acc: 0.6444 - val_loss: 0.8825 - val_acc: 0.6785
    Epoch 108/200
    15378/15378 [==============================] - 0s - loss: 0.8845 - acc: 0.6457 - val_loss: 0.8855 - val_acc: 0.7020
    Epoch 109/200
    15378/15378 [==============================] - 0s - loss: 0.8806 - acc: 0.6490 - val_loss: 0.8782 - val_acc: 0.6973
    Epoch 110/200
    15378/15378 [==============================] - 0s - loss: 0.8815 - acc: 0.6471 - val_loss: 0.8807 - val_acc: 0.6973
    Epoch 111/200
    15378/15378 [==============================] - 0s - loss: 0.8833 - acc: 0.6472 - val_loss: 0.8798 - val_acc: 0.6970
    Epoch 112/200
    15378/15378 [==============================] - 0s - loss: 0.8763 - acc: 0.6466 - val_loss: 0.8780 - val_acc: 0.6949
    Epoch 113/200
    15378/15378 [==============================] - 0s - loss: 0.8729 - acc: 0.6533 - val_loss: 0.8862 - val_acc: 0.6778
    Epoch 114/200
    15378/15378 [==============================] - 0s - loss: 0.8777 - acc: 0.6540 - val_loss: 0.8804 - val_acc: 0.7014
    Epoch 115/200
    15378/15378 [==============================] - 0s - loss: 0.8744 - acc: 0.6488 - val_loss: 0.8843 - val_acc: 0.7017
    Epoch 116/200
    15378/15378 [==============================] - 0s - loss: 0.8767 - acc: 0.6507 - val_loss: 0.8754 - val_acc: 0.6895
    Epoch 117/200
    15378/15378 [==============================] - 0s - loss: 0.8833 - acc: 0.6485 - val_loss: 0.8749 - val_acc: 0.6798
    Epoch 118/200
    15378/15378 [==============================] - 0s - loss: 0.8802 - acc: 0.6462 - val_loss: 0.8719 - val_acc: 0.6915
    Epoch 119/200
    15378/15378 [==============================] - 0s - loss: 0.8801 - acc: 0.6443 - val_loss: 0.8760 - val_acc: 0.6991
    Epoch 120/200
    15378/15378 [==============================] - 0s - loss: 0.8670 - acc: 0.6578 - val_loss: 0.8948 - val_acc: 0.6752
    Epoch 121/200
    15378/15378 [==============================] - 0s - loss: 0.8775 - acc: 0.6450 - val_loss: 0.8822 - val_acc: 0.6921
    Epoch 122/200
    15378/15378 [==============================] - 0s - loss: 0.8708 - acc: 0.6515 - val_loss: 0.8727 - val_acc: 0.6895
    Epoch 123/200
    15378/15378 [==============================] - 0s - loss: 0.8698 - acc: 0.6515 - val_loss: 0.8698 - val_acc: 0.6897
    Epoch 124/200
    15378/15378 [==============================] - 0s - loss: 0.8759 - acc: 0.6540 - val_loss: 0.8661 - val_acc: 0.6936
    Epoch 125/200
    15378/15378 [==============================] - 0s - loss: 0.8641 - acc: 0.6550 - val_loss: 0.8686 - val_acc: 0.7007
    Epoch 126/200
    15378/15378 [==============================] - 0s - loss: 0.8752 - acc: 0.6528 - val_loss: 0.8728 - val_acc: 0.7017
    Epoch 127/200
    15378/15378 [==============================] - 0s - loss: 0.8659 - acc: 0.6550 - val_loss: 0.8731 - val_acc: 0.6793
    Epoch 128/200
    15378/15378 [==============================] - 0s - loss: 0.8650 - acc: 0.6513 - val_loss: 0.8644 - val_acc: 0.7009
    Epoch 129/200
    15378/15378 [==============================] - 0s - loss: 0.8682 - acc: 0.6530 - val_loss: 0.8671 - val_acc: 0.6921
    Epoch 130/200
    15378/15378 [==============================] - 0s - loss: 0.8673 - acc: 0.6528 - val_loss: 0.8667 - val_acc: 0.7009
    Epoch 131/200
    15378/15378 [==============================] - 0s - loss: 0.8637 - acc: 0.6570 - val_loss: 0.8679 - val_acc: 0.7004
    Epoch 132/200
    15378/15378 [==============================] - 0s - loss: 0.8625 - acc: 0.6587 - val_loss: 0.8677 - val_acc: 0.6941
    Epoch 133/200
    15378/15378 [==============================] - 0s - loss: 0.8666 - acc: 0.6592 - val_loss: 0.8644 - val_acc: 0.7009
    Epoch 134/200
    15378/15378 [==============================] - 0s - loss: 0.8608 - acc: 0.6572 - val_loss: 0.8663 - val_acc: 0.6936
    Epoch 135/200
    15378/15378 [==============================] - 0s - loss: 0.8648 - acc: 0.6564 - val_loss: 0.8669 - val_acc: 0.7038
    Epoch 136/200
    15378/15378 [==============================] - 0s - loss: 0.8599 - acc: 0.6606 - val_loss: 0.8650 - val_acc: 0.7048
    Epoch 137/200
    15378/15378 [==============================] - 0s - loss: 0.8541 - acc: 0.6673 - val_loss: 0.8645 - val_acc: 0.7040
    Epoch 138/200
    15378/15378 [==============================] - 0s - loss: 0.8611 - acc: 0.6587 - val_loss: 0.8626 - val_acc: 0.7033
    Epoch 139/200
    15378/15378 [==============================] - 0s - loss: 0.8567 - acc: 0.6609 - val_loss: 0.8650 - val_acc: 0.7074
    Epoch 140/200
    15378/15378 [==============================] - 0s - loss: 0.8560 - acc: 0.6617 - val_loss: 0.8584 - val_acc: 0.6910
    Epoch 141/200
    15378/15378 [==============================] - 0s - loss: 0.8532 - acc: 0.6648 - val_loss: 0.8583 - val_acc: 0.6999
    Epoch 142/200
    15378/15378 [==============================] - 0s - loss: 0.8569 - acc: 0.6646 - val_loss: 0.8592 - val_acc: 0.6926
    Epoch 143/200
    15378/15378 [==============================] - 0s - loss: 0.8529 - acc: 0.6627 - val_loss: 0.8649 - val_acc: 0.7033
    Epoch 144/200
    15378/15378 [==============================] - 0s - loss: 0.8608 - acc: 0.6585 - val_loss: 0.8699 - val_acc: 0.7025
    Epoch 145/200
    15378/15378 [==============================] - 0s - loss: 0.8508 - acc: 0.6624 - val_loss: 0.8570 - val_acc: 0.7012
    Epoch 146/200
    15378/15378 [==============================] - 0s - loss: 0.8519 - acc: 0.6595 - val_loss: 0.8622 - val_acc: 0.7061
    Epoch 147/200
    15378/15378 [==============================] - 0s - loss: 0.8482 - acc: 0.6626 - val_loss: 0.8541 - val_acc: 0.7033
    Epoch 148/200
    15378/15378 [==============================] - 0s - loss: 0.8506 - acc: 0.6655 - val_loss: 0.8560 - val_acc: 0.6908
    Epoch 149/200
    15378/15378 [==============================] - 0s - loss: 0.8486 - acc: 0.6658 - val_loss: 0.8547 - val_acc: 0.6939
    Epoch 150/200
    15378/15378 [==============================] - 0s - loss: 0.8485 - acc: 0.6641 - val_loss: 0.8572 - val_acc: 0.6952
    Epoch 151/200
    15378/15378 [==============================] - 0s - loss: 0.8521 - acc: 0.6596 - val_loss: 0.8555 - val_acc: 0.7048
    Epoch 152/200
    15378/15378 [==============================] - 0s - loss: 0.8428 - acc: 0.6680 - val_loss: 0.8562 - val_acc: 0.7072
    Epoch 153/200
    15378/15378 [==============================] - 0s - loss: 0.8444 - acc: 0.6680 - val_loss: 0.8533 - val_acc: 0.7033
    Epoch 154/200
    15378/15378 [==============================] - 0s - loss: 0.8443 - acc: 0.6680 - val_loss: 0.8535 - val_acc: 0.7053
    Epoch 155/200
    15378/15378 [==============================] - 0s - loss: 0.8453 - acc: 0.6671 - val_loss: 0.8508 - val_acc: 0.7001
    Epoch 156/200
    15378/15378 [==============================] - 0s - loss: 0.8516 - acc: 0.6630 - val_loss: 0.8522 - val_acc: 0.6954
    Epoch 157/200
    15378/15378 [==============================] - 0s - loss: 0.8437 - acc: 0.6674 - val_loss: 0.8518 - val_acc: 0.7004
    Epoch 158/200
    15378/15378 [==============================] - 0s - loss: 0.8409 - acc: 0.6642 - val_loss: 0.8513 - val_acc: 0.6970
    Epoch 159/200
    15378/15378 [==============================] - 0s - loss: 0.8419 - acc: 0.6687 - val_loss: 0.8493 - val_acc: 0.7048
    Epoch 160/200
    15378/15378 [==============================] - 0s - loss: 0.8467 - acc: 0.6635 - val_loss: 0.8508 - val_acc: 0.6999
    Epoch 161/200
    15378/15378 [==============================] - 0s - loss: 0.8412 - acc: 0.6641 - val_loss: 0.8534 - val_acc: 0.6996
    Epoch 162/200
    15378/15378 [==============================] - 0s - loss: 0.8505 - acc: 0.6643 - val_loss: 0.8500 - val_acc: 0.6980
    Epoch 163/200
    15378/15378 [==============================] - 0s - loss: 0.8427 - acc: 0.6691 - val_loss: 0.8482 - val_acc: 0.7025
    Epoch 164/200
    15378/15378 [==============================] - 0s - loss: 0.8317 - acc: 0.6725 - val_loss: 0.8471 - val_acc: 0.7051
    Epoch 165/200
    15378/15378 [==============================] - 0s - loss: 0.8399 - acc: 0.6697 - val_loss: 0.8471 - val_acc: 0.7043
    Epoch 166/200
    15378/15378 [==============================] - 0s - loss: 0.8363 - acc: 0.6684 - val_loss: 0.8507 - val_acc: 0.7059
    Epoch 167/200
    15378/15378 [==============================] - 0s - loss: 0.8464 - acc: 0.6672 - val_loss: 0.8448 - val_acc: 0.6999
    Epoch 168/200
    15378/15378 [==============================] - 0s - loss: 0.8461 - acc: 0.6670 - val_loss: 0.8479 - val_acc: 0.7053
    Epoch 169/200
    15378/15378 [==============================] - 0s - loss: 0.8389 - acc: 0.6676 - val_loss: 0.8427 - val_acc: 0.7033
    Epoch 170/200
    15378/15378 [==============================] - 0s - loss: 0.8347 - acc: 0.6735 - val_loss: 0.8465 - val_acc: 0.7046
    Epoch 171/200
    15378/15378 [==============================] - 0s - loss: 0.8371 - acc: 0.6681 - val_loss: 0.8456 - val_acc: 0.7025
    Epoch 172/200
    15378/15378 [==============================] - 0s - loss: 0.8290 - acc: 0.6766 - val_loss: 0.8459 - val_acc: 0.6993
    Epoch 173/200
    15378/15378 [==============================] - 0s - loss: 0.8405 - acc: 0.6708 - val_loss: 0.8450 - val_acc: 0.7022
    Epoch 174/200
    15378/15378 [==============================] - 0s - loss: 0.8373 - acc: 0.6678 - val_loss: 0.8439 - val_acc: 0.7040
    Epoch 175/200
    15378/15378 [==============================] - 0s - loss: 0.8379 - acc: 0.6713 - val_loss: 0.8524 - val_acc: 0.6993
    Epoch 176/200
    15378/15378 [==============================] - 0s - loss: 0.8353 - acc: 0.6719 - val_loss: 0.8413 - val_acc: 0.7056
    Epoch 177/200
    15378/15378 [==============================] - 0s - loss: 0.8358 - acc: 0.6706 - val_loss: 0.8428 - val_acc: 0.7059
    Epoch 178/200
    15378/15378 [==============================] - 0s - loss: 0.8286 - acc: 0.6702 - val_loss: 0.8450 - val_acc: 0.7025
    Epoch 179/200
    15378/15378 [==============================] - 0s - loss: 0.8320 - acc: 0.6757 - val_loss: 0.8430 - val_acc: 0.7064
    Epoch 180/200
    15378/15378 [==============================] - 0s - loss: 0.8221 - acc: 0.6758 - val_loss: 0.8424 - val_acc: 0.6999
    Epoch 181/200
    15378/15378 [==============================] - 0s - loss: 0.8393 - acc: 0.6714 - val_loss: 0.8366 - val_acc: 0.7022
    Epoch 182/200
    15378/15378 [==============================] - 0s - loss: 0.8271 - acc: 0.6724 - val_loss: 0.8409 - val_acc: 0.7046
    Epoch 183/200
    15378/15378 [==============================] - 0s - loss: 0.8377 - acc: 0.6756 - val_loss: 0.8366 - val_acc: 0.6980
    Epoch 184/200
    15378/15378 [==============================] - 0s - loss: 0.8305 - acc: 0.6748 - val_loss: 0.8376 - val_acc: 0.6944
    Epoch 185/200
    15378/15378 [==============================] - 0s - loss: 0.8297 - acc: 0.6742 - val_loss: 0.8389 - val_acc: 0.7030
    Epoch 186/200
    15378/15378 [==============================] - 0s - loss: 0.8291 - acc: 0.6758 - val_loss: 0.8404 - val_acc: 0.7043
    Epoch 187/200
    15378/15378 [==============================] - 0s - loss: 0.8264 - acc: 0.6741 - val_loss: 0.8442 - val_acc: 0.7009
    Epoch 188/200
    15378/15378 [==============================] - 0s - loss: 0.8163 - acc: 0.6814 - val_loss: 0.8387 - val_acc: 0.7007
    Epoch 189/200
    15378/15378 [==============================] - 0s - loss: 0.8237 - acc: 0.6767 - val_loss: 0.8396 - val_acc: 0.7048
    Epoch 190/200
    15378/15378 [==============================] - 0s - loss: 0.8267 - acc: 0.6777 - val_loss: 0.8359 - val_acc: 0.7056
    Epoch 191/200
    15378/15378 [==============================] - 0s - loss: 0.8285 - acc: 0.6765 - val_loss: 0.8482 - val_acc: 0.6970
    Epoch 192/200
    15378/15378 [==============================] - 0s - loss: 0.8233 - acc: 0.6774 - val_loss: 0.8391 - val_acc: 0.7022
    Epoch 193/200
    15378/15378 [==============================] - 0s - loss: 0.8238 - acc: 0.6794 - val_loss: 0.8343 - val_acc: 0.7035
    Epoch 194/200
    15378/15378 [==============================] - 0s - loss: 0.8192 - acc: 0.6754 - val_loss: 0.8379 - val_acc: 0.7038
    Epoch 195/200
    15378/15378 [==============================] - 0s - loss: 0.8210 - acc: 0.6783 - val_loss: 0.8350 - val_acc: 0.7012
    Epoch 196/200
    15378/15378 [==============================] - 0s - loss: 0.8284 - acc: 0.6766 - val_loss: 0.8364 - val_acc: 0.7022
    Epoch 197/200
    15378/15378 [==============================] - 0s - loss: 0.8238 - acc: 0.6776 - val_loss: 0.8444 - val_acc: 0.7009
    Epoch 198/200
    15378/15378 [==============================] - 0s - loss: 0.8159 - acc: 0.6772 - val_loss: 0.8341 - val_acc: 0.7020
    Epoch 199/200
    15378/15378 [==============================] - 0s - loss: 0.8252 - acc: 0.6816 - val_loss: 0.8349 - val_acc: 0.7038
    Epoch 200/200
    15378/15378 [==============================] - 0s - loss: 0.8190 - acc: 0.6767 - val_loss: 0.8367 - val_acc: 0.7061





    <keras.callbacks.History at 0x143ce7ac8>




```python
model.test_on_batch(X_test, y_test, sample_weight=None)
```




    [0.74140817, 0.69601679]



The model performs well both for the validation and the test sets (65% might not seem a lot, but it is satisfying enough for our problem, where all the labels are very subjective (Was Larry Bird a "SM-PF" or a "PF-SF"? Nobody can tell).

Now we train again the model, using all the training data (we will still reserve the 2017 season out of the training).


```python
# Production model, using all data
model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0, verbose=1)
```

    Epoch 1/200
    19223/19223 [==============================] - 0s - loss: 0.8416 - acc: 0.6743     
    Epoch 2/200
    19223/19223 [==============================] - 0s - loss: 0.8484 - acc: 0.6783     
    Epoch 3/200
    19223/19223 [==============================] - 0s - loss: 0.8460 - acc: 0.6733     
    Epoch 4/200
    19223/19223 [==============================] - 0s - loss: 0.8332 - acc: 0.6786     
    Epoch 5/200
    19223/19223 [==============================] - 0s - loss: 0.8373 - acc: 0.6799     
    Epoch 6/200
    19223/19223 [==============================] - 0s - loss: 0.8416 - acc: 0.6742     
    Epoch 7/200
    19223/19223 [==============================] - 0s - loss: 0.8364 - acc: 0.6770     
    Epoch 8/200
    19223/19223 [==============================] - 0s - loss: 0.8407 - acc: 0.6744     
    Epoch 9/200
    19223/19223 [==============================] - 0s - loss: 0.8351 - acc: 0.6754     
    Epoch 10/200
    19223/19223 [==============================] - 0s - loss: 0.8304 - acc: 0.6786     
    Epoch 11/200
    19223/19223 [==============================] - 0s - loss: 0.8369 - acc: 0.6747     
    Epoch 12/200
    19223/19223 [==============================] - 0s - loss: 0.8309 - acc: 0.6798     
    Epoch 13/200
    19223/19223 [==============================] - 0s - loss: 0.8369 - acc: 0.6813     
    Epoch 14/200
    19223/19223 [==============================] - 0s - loss: 0.8351 - acc: 0.6779     
    Epoch 15/200
    19223/19223 [==============================] - 0s - loss: 0.8272 - acc: 0.6858     
    Epoch 16/200
    19223/19223 [==============================] - 0s - loss: 0.8340 - acc: 0.6770     
    Epoch 17/200
    19223/19223 [==============================] - 0s - loss: 0.8279 - acc: 0.6789     
    Epoch 18/200
    19223/19223 [==============================] - 0s - loss: 0.8252 - acc: 0.6788     
    Epoch 19/200
    19223/19223 [==============================] - 0s - loss: 0.8297 - acc: 0.6768     
    Epoch 20/200
    19223/19223 [==============================] - 0s - loss: 0.8257 - acc: 0.6797     
    Epoch 21/200
    19223/19223 [==============================] - 0s - loss: 0.8311 - acc: 0.6812     
    Epoch 22/200
    19223/19223 [==============================] - 0s - loss: 0.8262 - acc: 0.6822     
    Epoch 23/200
    19223/19223 [==============================] - 0s - loss: 0.8265 - acc: 0.6794     
    Epoch 24/200
    19223/19223 [==============================] - 0s - loss: 0.8307 - acc: 0.6787     
    Epoch 25/200
    19223/19223 [==============================] - 0s - loss: 0.8188 - acc: 0.6855     
    Epoch 26/200
    19223/19223 [==============================] - 0s - loss: 0.8270 - acc: 0.6812     
    Epoch 27/200
    19223/19223 [==============================] - 0s - loss: 0.8277 - acc: 0.6815     
    Epoch 28/200
    19223/19223 [==============================] - 0s - loss: 0.8240 - acc: 0.6826     
    Epoch 29/200
    19223/19223 [==============================] - 0s - loss: 0.8298 - acc: 0.6829     
    Epoch 30/200
    19223/19223 [==============================] - 0s - loss: 0.8245 - acc: 0.6829     
    Epoch 31/200
    19223/19223 [==============================] - 0s - loss: 0.8164 - acc: 0.6847     
    Epoch 32/200
    19223/19223 [==============================] - 0s - loss: 0.8213 - acc: 0.6819     
    Epoch 33/200
    19223/19223 [==============================] - 0s - loss: 0.8260 - acc: 0.6824     
    Epoch 34/200
    19223/19223 [==============================] - 0s - loss: 0.8271 - acc: 0.6756     
    Epoch 35/200
    19223/19223 [==============================] - 0s - loss: 0.8208 - acc: 0.6854     
    Epoch 36/200
    19223/19223 [==============================] - 0s - loss: 0.8185 - acc: 0.6883     
    Epoch 37/200
    19223/19223 [==============================] - 0s - loss: 0.8201 - acc: 0.6878     
    Epoch 38/200
    19223/19223 [==============================] - 0s - loss: 0.8208 - acc: 0.6858     
    Epoch 39/200
    19223/19223 [==============================] - 0s - loss: 0.8143 - acc: 0.6884     
    Epoch 40/200
    19223/19223 [==============================] - 0s - loss: 0.8136 - acc: 0.6872     
    Epoch 41/200
    19223/19223 [==============================] - 0s - loss: 0.8187 - acc: 0.6823     
    Epoch 42/200
    19223/19223 [==============================] - 0s - loss: 0.8139 - acc: 0.6887     
    Epoch 43/200
    19223/19223 [==============================] - 0s - loss: 0.8195 - acc: 0.6856     
    Epoch 44/200
    19223/19223 [==============================] - 0s - loss: 0.8131 - acc: 0.6856     
    Epoch 45/200
    19223/19223 [==============================] - 0s - loss: 0.8148 - acc: 0.6895     
    Epoch 46/200
    19223/19223 [==============================] - 0s - loss: 0.8097 - acc: 0.6878     
    Epoch 47/200
    19223/19223 [==============================] - 0s - loss: 0.8165 - acc: 0.6894     
    Epoch 48/200
    19223/19223 [==============================] - 0s - loss: 0.8104 - acc: 0.6870     
    Epoch 49/200
    19223/19223 [==============================] - 0s - loss: 0.8102 - acc: 0.6891     
    Epoch 50/200
    19223/19223 [==============================] - 0s - loss: 0.8149 - acc: 0.6853     
    Epoch 51/200
    19223/19223 [==============================] - 0s - loss: 0.8129 - acc: 0.6902     
    Epoch 52/200
    19223/19223 [==============================] - 0s - loss: 0.8149 - acc: 0.6908     
    Epoch 53/200
    19223/19223 [==============================] - 0s - loss: 0.8091 - acc: 0.6897     
    Epoch 54/200
    19223/19223 [==============================] - 0s - loss: 0.8109 - acc: 0.6866     
    Epoch 55/200
    19223/19223 [==============================] - 0s - loss: 0.8142 - acc: 0.6863     
    Epoch 56/200
    19223/19223 [==============================] - 0s - loss: 0.8082 - acc: 0.6923     
    Epoch 57/200
    19223/19223 [==============================] - 0s - loss: 0.8060 - acc: 0.6877     
    Epoch 58/200
    19223/19223 [==============================] - 0s - loss: 0.8142 - acc: 0.6891     
    Epoch 59/200
    19223/19223 [==============================] - 0s - loss: 0.8087 - acc: 0.6897     
    Epoch 60/200
    19223/19223 [==============================] - 0s - loss: 0.8084 - acc: 0.6917     
    Epoch 61/200
    19223/19223 [==============================] - 0s - loss: 0.8087 - acc: 0.6921     
    Epoch 62/200
    19223/19223 [==============================] - 0s - loss: 0.8139 - acc: 0.6927     
    Epoch 63/200
    19223/19223 [==============================] - 0s - loss: 0.8085 - acc: 0.6892     
    Epoch 64/200
    19223/19223 [==============================] - 0s - loss: 0.8044 - acc: 0.6909     
    Epoch 65/200
    19223/19223 [==============================] - 0s - loss: 0.8056 - acc: 0.6879     
    Epoch 66/200
    19223/19223 [==============================] - 0s - loss: 0.8142 - acc: 0.6848     
    Epoch 67/200
    19223/19223 [==============================] - 0s - loss: 0.8023 - acc: 0.6923     
    Epoch 68/200
    19223/19223 [==============================] - 0s - loss: 0.8074 - acc: 0.6917     - ETA: 0s - loss: 0.7954 - acc:
    Epoch 69/200
    19223/19223 [==============================] - 0s - loss: 0.8090 - acc: 0.6919     
    Epoch 70/200
    19223/19223 [==============================] - 0s - loss: 0.8015 - acc: 0.6907     
    Epoch 71/200
    19223/19223 [==============================] - 0s - loss: 0.8039 - acc: 0.6923     
    Epoch 72/200
    19223/19223 [==============================] - 0s - loss: 0.7997 - acc: 0.6933     
    Epoch 73/200
    19223/19223 [==============================] - 0s - loss: 0.8000 - acc: 0.6961     
    Epoch 74/200
    19223/19223 [==============================] - 0s - loss: 0.8045 - acc: 0.6940     
    Epoch 75/200
    19223/19223 [==============================] - 0s - loss: 0.8059 - acc: 0.6954     
    Epoch 76/200
    19223/19223 [==============================] - 0s - loss: 0.8064 - acc: 0.6921     
    Epoch 77/200
    19223/19223 [==============================] - 0s - loss: 0.8043 - acc: 0.6923     
    Epoch 78/200
    19223/19223 [==============================] - 0s - loss: 0.8091 - acc: 0.6900     
    Epoch 79/200
    19223/19223 [==============================] - 0s - loss: 0.8057 - acc: 0.6947     
    Epoch 80/200
    19223/19223 [==============================] - 0s - loss: 0.8047 - acc: 0.6965     
    Epoch 81/200
    19223/19223 [==============================] - 0s - loss: 0.8037 - acc: 0.6958     
    Epoch 82/200
    19223/19223 [==============================] - 0s - loss: 0.8008 - acc: 0.6949     
    Epoch 83/200
    19223/19223 [==============================] - 0s - loss: 0.8002 - acc: 0.6954     
    Epoch 84/200
    19223/19223 [==============================] - 0s - loss: 0.8002 - acc: 0.6980     
    Epoch 85/200
    19223/19223 [==============================] - 0s - loss: 0.8018 - acc: 0.6954     
    Epoch 86/200
    19223/19223 [==============================] - 0s - loss: 0.8030 - acc: 0.6973     
    Epoch 87/200
    19223/19223 [==============================] - 0s - loss: 0.7959 - acc: 0.6958     
    Epoch 88/200
    19223/19223 [==============================] - 0s - loss: 0.8052 - acc: 0.6933     
    Epoch 89/200
    19223/19223 [==============================] - 0s - loss: 0.8008 - acc: 0.6944     
    Epoch 90/200
    19223/19223 [==============================] - 0s - loss: 0.7945 - acc: 0.6982     
    Epoch 91/200
    19223/19223 [==============================] - 0s - loss: 0.8060 - acc: 0.6917     
    Epoch 92/200
    19223/19223 [==============================] - 0s - loss: 0.8007 - acc: 0.6935     
    Epoch 93/200
    19223/19223 [==============================] - 0s - loss: 0.7936 - acc: 0.6942     
    Epoch 94/200
    19223/19223 [==============================] - 0s - loss: 0.7949 - acc: 0.7004     
    Epoch 95/200
    19223/19223 [==============================] - 0s - loss: 0.7981 - acc: 0.6956     
    Epoch 96/200
    19223/19223 [==============================] - 0s - loss: 0.7958 - acc: 0.6986     
    Epoch 97/200
    19223/19223 [==============================] - 0s - loss: 0.8041 - acc: 0.6958     
    Epoch 98/200
    19223/19223 [==============================] - 0s - loss: 0.7995 - acc: 0.6931     
    Epoch 99/200
    19223/19223 [==============================] - 0s - loss: 0.7982 - acc: 0.6955     
    Epoch 100/200
    19223/19223 [==============================] - 0s - loss: 0.7981 - acc: 0.6960     
    Epoch 101/200
    19223/19223 [==============================] - 0s - loss: 0.8055 - acc: 0.6947     
    Epoch 102/200
    19223/19223 [==============================] - 0s - loss: 0.7979 - acc: 0.6946     
    Epoch 103/200
    19223/19223 [==============================] - 0s - loss: 0.7952 - acc: 0.6972     
    Epoch 104/200
    19223/19223 [==============================] - 0s - loss: 0.7964 - acc: 0.6949     
    Epoch 105/200
    19223/19223 [==============================] - 0s - loss: 0.7938 - acc: 0.6988     
    Epoch 106/200
    19223/19223 [==============================] - 0s - loss: 0.7964 - acc: 0.6993     
    Epoch 107/200
    19223/19223 [==============================] - 0s - loss: 0.7940 - acc: 0.6979     
    Epoch 108/200
    19223/19223 [==============================] - 0s - loss: 0.7915 - acc: 0.6968     
    Epoch 109/200
    19223/19223 [==============================] - 0s - loss: 0.7930 - acc: 0.6933     
    Epoch 110/200
    19223/19223 [==============================] - 0s - loss: 0.7858 - acc: 0.7012     
    Epoch 111/200
    19223/19223 [==============================] - 0s - loss: 0.7945 - acc: 0.6983     
    Epoch 112/200
    19223/19223 [==============================] - 0s - loss: 0.7922 - acc: 0.7011     
    Epoch 113/200
    19223/19223 [==============================] - 0s - loss: 0.7932 - acc: 0.6957     
    Epoch 114/200
    19223/19223 [==============================] - 0s - loss: 0.7935 - acc: 0.6948     
    Epoch 115/200
    19223/19223 [==============================] - 0s - loss: 0.7905 - acc: 0.6966     
    Epoch 116/200
    19223/19223 [==============================] - 0s - loss: 0.7937 - acc: 0.6995     
    Epoch 117/200
    19223/19223 [==============================] - 0s - loss: 0.7883 - acc: 0.6948     
    Epoch 118/200
    19223/19223 [==============================] - 0s - loss: 0.7888 - acc: 0.6979     
    Epoch 119/200
    19223/19223 [==============================] - 0s - loss: 0.7874 - acc: 0.7022     
    Epoch 120/200
    19223/19223 [==============================] - 0s - loss: 0.7880 - acc: 0.6998     
    Epoch 121/200
    19223/19223 [==============================] - 0s - loss: 0.7965 - acc: 0.7020     
    Epoch 122/200
    19223/19223 [==============================] - 0s - loss: 0.7864 - acc: 0.6978     
    Epoch 123/200
    19223/19223 [==============================] - 0s - loss: 0.7895 - acc: 0.7021     
    Epoch 124/200
    19223/19223 [==============================] - 0s - loss: 0.7886 - acc: 0.6989     
    Epoch 125/200
    19223/19223 [==============================] - 0s - loss: 0.7880 - acc: 0.7000     
    Epoch 126/200
    19223/19223 [==============================] - 0s - loss: 0.7841 - acc: 0.7047     
    Epoch 127/200
    19223/19223 [==============================] - 0s - loss: 0.7942 - acc: 0.7020     - ETA: 0s - loss: 0.7756 - acc:
    Epoch 128/200
    19223/19223 [==============================] - 0s - loss: 0.7894 - acc: 0.7009     
    Epoch 129/200
    19223/19223 [==============================] - 0s - loss: 0.7955 - acc: 0.6960     
    Epoch 130/200
    19223/19223 [==============================] - 0s - loss: 0.7852 - acc: 0.7029     
    Epoch 131/200
    19223/19223 [==============================] - 0s - loss: 0.7895 - acc: 0.6993     
    Epoch 132/200
    19223/19223 [==============================] - 0s - loss: 0.7914 - acc: 0.7016     
    Epoch 133/200
    19223/19223 [==============================] - 0s - loss: 0.7862 - acc: 0.7040     
    Epoch 134/200
    19223/19223 [==============================] - 0s - loss: 0.7888 - acc: 0.6985     
    Epoch 135/200
    19223/19223 [==============================] - 0s - loss: 0.7904 - acc: 0.7014     
    Epoch 136/200
    19223/19223 [==============================] - 0s - loss: 0.7891 - acc: 0.7045     
    Epoch 137/200
    19223/19223 [==============================] - 0s - loss: 0.7838 - acc: 0.7069     
    Epoch 138/200
    19223/19223 [==============================] - 0s - loss: 0.7848 - acc: 0.7025     
    Epoch 139/200
    19223/19223 [==============================] - 0s - loss: 0.7844 - acc: 0.6996     
    Epoch 140/200
    19223/19223 [==============================] - 0s - loss: 0.7818 - acc: 0.7033     
    Epoch 141/200
    19223/19223 [==============================] - 0s - loss: 0.7872 - acc: 0.7033     
    Epoch 142/200
    19223/19223 [==============================] - 0s - loss: 0.7895 - acc: 0.6995     
    Epoch 143/200
    19223/19223 [==============================] - 0s - loss: 0.7851 - acc: 0.7052     
    Epoch 144/200
    19223/19223 [==============================] - 0s - loss: 0.7809 - acc: 0.7044     
    Epoch 145/200
    19223/19223 [==============================] - 0s - loss: 0.7852 - acc: 0.7037     
    Epoch 146/200
    19223/19223 [==============================] - 0s - loss: 0.7904 - acc: 0.6993     
    Epoch 147/200
    19223/19223 [==============================] - 0s - loss: 0.7879 - acc: 0.7000     
    Epoch 148/200
    19223/19223 [==============================] - 0s - loss: 0.7868 - acc: 0.6990     
    Epoch 149/200
    19223/19223 [==============================] - 0s - loss: 0.7870 - acc: 0.7015     
    Epoch 150/200
    19223/19223 [==============================] - 0s - loss: 0.7812 - acc: 0.7029     
    Epoch 151/200
    19223/19223 [==============================] - 0s - loss: 0.7866 - acc: 0.7037     
    Epoch 152/200
    19223/19223 [==============================] - 0s - loss: 0.7857 - acc: 0.7041     
    Epoch 153/200
    19223/19223 [==============================] - 0s - loss: 0.7875 - acc: 0.7049     
    Epoch 154/200
    19223/19223 [==============================] - 0s - loss: 0.7785 - acc: 0.7028     
    Epoch 155/200
    19223/19223 [==============================] - 0s - loss: 0.7807 - acc: 0.7051     
    Epoch 156/200
    19223/19223 [==============================] - 0s - loss: 0.7920 - acc: 0.6994     
    Epoch 157/200
    19223/19223 [==============================] - 0s - loss: 0.7910 - acc: 0.6988     
    Epoch 158/200
    19223/19223 [==============================] - 0s - loss: 0.7842 - acc: 0.7042     
    Epoch 159/200
    19223/19223 [==============================] - 0s - loss: 0.7837 - acc: 0.7012     
    Epoch 160/200
    19223/19223 [==============================] - 0s - loss: 0.7842 - acc: 0.7037     
    Epoch 161/200
    19223/19223 [==============================] - 0s - loss: 0.7887 - acc: 0.7041     - ETA: 0s - loss: 0.7762 - ac
    Epoch 162/200
    19223/19223 [==============================] - 0s - loss: 0.7814 - acc: 0.7050     
    Epoch 163/200
    19223/19223 [==============================] - 0s - loss: 0.7767 - acc: 0.7056     
    Epoch 164/200
    19223/19223 [==============================] - 0s - loss: 0.7811 - acc: 0.7050     
    Epoch 165/200
    19223/19223 [==============================] - 0s - loss: 0.7847 - acc: 0.7060     
    Epoch 166/200
    19223/19223 [==============================] - 0s - loss: 0.7828 - acc: 0.7025     
    Epoch 167/200
    19223/19223 [==============================] - 0s - loss: 0.7825 - acc: 0.7027     
    Epoch 168/200
    19223/19223 [==============================] - 0s - loss: 0.7768 - acc: 0.7063     
    Epoch 169/200
    19223/19223 [==============================] - 0s - loss: 0.7809 - acc: 0.7037     
    Epoch 170/200
    19223/19223 [==============================] - 0s - loss: 0.7836 - acc: 0.7021     
    Epoch 171/200
    19223/19223 [==============================] - 0s - loss: 0.7808 - acc: 0.7041     
    Epoch 172/200
    19223/19223 [==============================] - 0s - loss: 0.7832 - acc: 0.7043     
    Epoch 173/200
    19223/19223 [==============================] - 0s - loss: 0.7802 - acc: 0.7022     
    Epoch 174/200
    19223/19223 [==============================] - 0s - loss: 0.7789 - acc: 0.7025     
    Epoch 175/200
    19223/19223 [==============================] - 0s - loss: 0.7811 - acc: 0.7045     
    Epoch 176/200
    19223/19223 [==============================] - 0s - loss: 0.7817 - acc: 0.7039     
    Epoch 177/200
    19223/19223 [==============================] - 0s - loss: 0.7835 - acc: 0.7069     
    Epoch 178/200
    19223/19223 [==============================] - 0s - loss: 0.7831 - acc: 0.7018     
    Epoch 179/200
    19223/19223 [==============================] - 0s - loss: 0.7807 - acc: 0.7034     
    Epoch 180/200
    19223/19223 [==============================] - 0s - loss: 0.7772 - acc: 0.7043     
    Epoch 181/200
    19223/19223 [==============================] - 0s - loss: 0.7825 - acc: 0.7029     
    Epoch 182/200
    19223/19223 [==============================] - 0s - loss: 0.7818 - acc: 0.7032     
    Epoch 183/200
    19223/19223 [==============================] - 0s - loss: 0.7784 - acc: 0.7052     
    Epoch 184/200
    19223/19223 [==============================] - 0s - loss: 0.7826 - acc: 0.7037     
    Epoch 185/200
    19223/19223 [==============================] - 0s - loss: 0.7722 - acc: 0.7071     
    Epoch 186/200
    19223/19223 [==============================] - 0s - loss: 0.7851 - acc: 0.7019     
    Epoch 187/200
    19223/19223 [==============================] - 0s - loss: 0.7755 - acc: 0.7103     
    Epoch 188/200
    19223/19223 [==============================] - 0s - loss: 0.7779 - acc: 0.7080     
    Epoch 189/200
    19223/19223 [==============================] - 0s - loss: 0.7805 - acc: 0.7024     
    Epoch 190/200
    19223/19223 [==============================] - 0s - loss: 0.7814 - acc: 0.7096     
    Epoch 191/200
    19223/19223 [==============================] - 0s - loss: 0.7835 - acc: 0.7054     
    Epoch 192/200
    19223/19223 [==============================] - 0s - loss: 0.7822 - acc: 0.7099     
    Epoch 193/200
    19223/19223 [==============================] - 0s - loss: 0.7804 - acc: 0.7048     
    Epoch 194/200
    19223/19223 [==============================] - 0s - loss: 0.7807 - acc: 0.7035     
    Epoch 195/200
    19223/19223 [==============================] - 0s - loss: 0.7800 - acc: 0.7105     
    Epoch 196/200
    19223/19223 [==============================] - 0s - loss: 0.7706 - acc: 0.7108     
    Epoch 197/200
    19223/19223 [==============================] - 0s - loss: 0.7739 - acc: 0.7042     
    Epoch 198/200
    19223/19223 [==============================] - 0s - loss: 0.7772 - acc: 0.7081     
    Epoch 199/200
    19223/19223 [==============================] - 0s - loss: 0.7785 - acc: 0.7067     
    Epoch 200/200
    19223/19223 [==============================] - 0s - loss: 0.7720 - acc: 0.7073     





    <keras.callbacks.History at 0x143ce7e48>



## Predicting the positions of the First NBA Team of 2017 ##

As a first test of the model, we will predict the positions of the player in the First NBA Team of 2017


```python
first_team_members = ['Russell Westbrook', 'James Harden', 'Anthony Davis', 'LeBron James', 'Kawhi Leonard']
first_team_stats = data[[((x[1]['Player'] in first_team_members) & (x[1]['Year']==2017)) for x in data.iterrows()]]
first_team_stats
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Player</th>
      <th>Pos</th>
      <th>Age</th>
      <th>G</th>
      <th>GS</th>
      <th>MP</th>
      <th>PER</th>
      <th>TS%</th>
      <th>3PAr</th>
      <th>...</th>
      <th>DRB</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>TOV</th>
      <th>PF</th>
      <th>PTS</th>
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19326</th>
      <td>2017.0</td>
      <td>Anthony Davis</td>
      <td>C</td>
      <td>23.0</td>
      <td>75.0</td>
      <td>75.0</td>
      <td>2708.0</td>
      <td>0.365583</td>
      <td>0.579</td>
      <td>0.088</td>
      <td>...</td>
      <td>9.465288</td>
      <td>11.778434</td>
      <td>2.087149</td>
      <td>1.249631</td>
      <td>2.220089</td>
      <td>2.406204</td>
      <td>2.233383</td>
      <td>27.903988</td>
      <td>206.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>19399</th>
      <td>2017.0</td>
      <td>James Harden</td>
      <td>PG</td>
      <td>27.0</td>
      <td>81.0</td>
      <td>81.0</td>
      <td>2947.0</td>
      <td>0.333492</td>
      <td>0.613</td>
      <td>0.493</td>
      <td>...</td>
      <td>6.889718</td>
      <td>8.050221</td>
      <td>11.067526</td>
      <td>1.465898</td>
      <td>0.451985</td>
      <td>5.668137</td>
      <td>2.626400</td>
      <td>28.780455</td>
      <td>196.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>19444</th>
      <td>2017.0</td>
      <td>LeBron James</td>
      <td>SF</td>
      <td>32.0</td>
      <td>74.0</td>
      <td>74.0</td>
      <td>2794.0</td>
      <td>0.347888</td>
      <td>0.619</td>
      <td>0.254</td>
      <td>...</td>
      <td>6.996421</td>
      <td>8.246242</td>
      <td>8.323550</td>
      <td>1.185397</td>
      <td>0.566929</td>
      <td>3.904080</td>
      <td>1.726557</td>
      <td>25.176807</td>
      <td>203.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>19486</th>
      <td>2017.0</td>
      <td>Kawhi Leonard</td>
      <td>SF</td>
      <td>25.0</td>
      <td>74.0</td>
      <td>74.0</td>
      <td>2474.0</td>
      <td>0.400162</td>
      <td>0.611</td>
      <td>0.294</td>
      <td>...</td>
      <td>5.092967</td>
      <td>6.257074</td>
      <td>3.783347</td>
      <td>1.920776</td>
      <td>0.800323</td>
      <td>2.240905</td>
      <td>1.775263</td>
      <td>27.472918</td>
      <td>201.0</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>19671</th>
      <td>2017.0</td>
      <td>Russell Westbrook</td>
      <td>PG</td>
      <td>28.0</td>
      <td>81.0</td>
      <td>81.0</td>
      <td>2802.0</td>
      <td>0.393148</td>
      <td>0.554</td>
      <td>0.300</td>
      <td>...</td>
      <td>9.340471</td>
      <td>11.100642</td>
      <td>10.792291</td>
      <td>1.708779</td>
      <td>0.398287</td>
      <td>5.627409</td>
      <td>2.441113</td>
      <td>32.865096</td>
      <td>190.0</td>
      <td>90.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 51 columns</p>
</div>




```python
pd.DataFrame(index=first_team_stats.loc[:, 'Player'].values, data={'Real': first_team_stats.loc[:, 'Pos'].values,
    'Predicted':encoder.inverse_transform(model.predict(Xnorm[first_team_stats.index, :]))})
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted</th>
      <th>Real</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Anthony Davis</th>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>James Harden</th>
      <td>PG</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>LeBron James</th>
      <td>PF</td>
      <td>SF</td>
    </tr>
    <tr>
      <th>Kawhi Leonard</th>
      <td>SF</td>
      <td>SF</td>
    </tr>
    <tr>
      <th>Russell Westbrook</th>
      <td>PG</td>
      <td>PG</td>
    </tr>
  </tbody>
</table>
</div>



The model gets right four of the five. It's even more interesting that the one that gets wrong, Anthony Davis, can play in both PF and C positions, and that in the last season, he played more as a Power Forward than as a Center, as the model predicts:

[New Orleans Pelicans Depth Chart - 2016-17](http://www.espn.com/nba/team/depth/_/name/no/new-orleans-pelicans).

## Predicting the positions of the NBA MVP ##

We will use now the model to predict the positions of all the NBA MVP since the creation of the award, in 1956.


```python
mvp = [(1956, 'Bob Pettit'), (1957, 'Bob Cousy'), (1958, 'Bill Russell'), (1959, 'Bob Pettit'), 
(1960, 'Wilt Chamberlain'), (1961, 'Bill Russell'), (1962, 'Bill Russell'), (1963, 'Bill Russell'),
(1964, 'Oscar Robertson'), (1965, 'Bill Russell'), (1966, 'Wilt Chamberlain'), (1967, 'Wilt Chamberlain'),
(1968, 'Wilt Chamberlain'), (1969, 'Wes Unseld'), (1970, 'Willis Reed'), (1971, 'Lew Alcindor'), 
(1972, 'Kareem Abdul-Jabbar'), (1973, 'Dave Cowens'), (19704, 'Kareem Abdul-Jabbar'), (1975, 'Bob McAdoo'),
(1976, 'Kareem Abdul-Jabbar'), (1977, 'Kareem Abdul-Jabbar'), (1978, 'Bill Walton'), (1979, 'Moses Malone'), 
(1980, 'Kareem Abdul-Jabbar'), (1981, 'Julius Erving'), (1982, 'Moses Malone'), (1983, 'Moses Malone'), 
(1984, 'Larry Bird'), (1985, 'Larry Bird'), (1986, 'Larry Bird'), (1987, 'Magic Johnson'), 
(1988, 'Michael Jordan'), (1989, 'Magic Johnson'), (1990, 'Magic Johnson'), (1991, 'Michael Jordan'),
(1992, 'Michael Jordan'), (1993, 'Charles Barkley'), (1994, 'Hakeem Olajuwon'), (1995, 'David Robinson'),  
(1996, 'Michael Jordan'), (1997, 'Karl Malone'), (1998, 'Michael Jordan'), (1999, 'Karl Malone'), 
(2000, 'Shaquille O\'Neal'), (2001, 'Allen Iverson'), (2002, 'Tim Duncan'), (2003, 'Tim Duncan'), 
(2004, 'Kevin Garnett'), (2005, 'Steve Nash'), (2006, 'Steve Nash'), (2007, 'Dirk Nowitzki'), 
(2008, 'Kobe Bryant'), (2009, 'LeBron James'), (2010, 'LeBron James'), (2011, 'Derrick Rose'), 
(2012, 'LeBron James'), (2013, 'LeBron James'), (2014, 'Kevin Durant'), (2015, 'Stephen Curry'),
(2016, 'Stephen Curry')]
```


```python
mvp_stats = pd.concat([data[(data['Player'] == x[1]) & (data['Year']==x[0])] for x in mvp], axis=0)
```


```python
mvp_stats
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Player</th>
      <th>Pos</th>
      <th>Age</th>
      <th>G</th>
      <th>GS</th>
      <th>MP</th>
      <th>PER</th>
      <th>TS%</th>
      <th>3PAr</th>
      <th>...</th>
      <th>DRB</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>TOV</th>
      <th>PF</th>
      <th>PTS</th>
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>497</th>
      <td>1956.0</td>
      <td>Bob Pettit</td>
      <td>C</td>
      <td>23.0</td>
      <td>72.0</td>
      <td>0.0</td>
      <td>2794.0</td>
      <td>0.351754</td>
      <td>0.502</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>14.997853</td>
      <td>2.435218</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.602720</td>
      <td>23.823908</td>
      <td>206.0</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>536</th>
      <td>1957.0</td>
      <td>Bob Cousy</td>
      <td>PG</td>
      <td>28.0</td>
      <td>64.0</td>
      <td>0.0</td>
      <td>2364.0</td>
      <td>0.319797</td>
      <td>0.452</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>4.705584</td>
      <td>7.279188</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.040609</td>
      <td>20.086294</td>
      <td>185.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>688</th>
      <td>1958.0</td>
      <td>Bill Russell</td>
      <td>C</td>
      <td>23.0</td>
      <td>69.0</td>
      <td>0.0</td>
      <td>2640.0</td>
      <td>0.310909</td>
      <td>0.465</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>21.327273</td>
      <td>2.754545</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.468182</td>
      <td>15.572727</td>
      <td>208.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>784</th>
      <td>1959.0</td>
      <td>Bob Pettit</td>
      <td>PF</td>
      <td>26.0</td>
      <td>72.0</td>
      <td>0.0</td>
      <td>2873.0</td>
      <td>0.353359</td>
      <td>0.519</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>14.810999</td>
      <td>2.769231</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.506091</td>
      <td>26.376610</td>
      <td>206.0</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>827</th>
      <td>1960.0</td>
      <td>Wilt Chamberlain</td>
      <td>C</td>
      <td>23.0</td>
      <td>72.0</td>
      <td>0.0</td>
      <td>3338.0</td>
      <td>0.301977</td>
      <td>0.493</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>20.933493</td>
      <td>1.811863</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.617735</td>
      <td>29.194727</td>
      <td>216.0</td>
      <td>124.0</td>
    </tr>
    <tr>
      <th>992</th>
      <td>1961.0</td>
      <td>Bill Russell</td>
      <td>C</td>
      <td>26.0</td>
      <td>78.0</td>
      <td>0.0</td>
      <td>3458.0</td>
      <td>0.188433</td>
      <td>0.454</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>19.447079</td>
      <td>2.790052</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.613650</td>
      <td>13.762869</td>
      <td>208.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>1962.0</td>
      <td>Bill Russell</td>
      <td>C</td>
      <td>27.0</td>
      <td>76.0</td>
      <td>0.0</td>
      <td>3433.0</td>
      <td>0.203437</td>
      <td>0.489</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>18.770754</td>
      <td>3.575881</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.170696</td>
      <td>15.058549</td>
      <td>208.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>1208</th>
      <td>1963.0</td>
      <td>Bill Russell</td>
      <td>C</td>
      <td>28.0</td>
      <td>78.0</td>
      <td>0.0</td>
      <td>3500.0</td>
      <td>0.187200</td>
      <td>0.464</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>18.956571</td>
      <td>3.579429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.944000</td>
      <td>13.464000</td>
      <td>208.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>1320</th>
      <td>1964.0</td>
      <td>Oscar Robertson</td>
      <td>PG</td>
      <td>25.0</td>
      <td>79.0</td>
      <td>0.0</td>
      <td>3559.0</td>
      <td>0.279180</td>
      <td>0.576</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>7.920202</td>
      <td>8.779994</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.832256</td>
      <td>25.085698</td>
      <td>196.0</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>1440</th>
      <td>1965.0</td>
      <td>Bill Russell</td>
      <td>C</td>
      <td>30.0</td>
      <td>78.0</td>
      <td>0.0</td>
      <td>3466.0</td>
      <td>0.202539</td>
      <td>0.472</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>19.506059</td>
      <td>4.258511</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.118869</td>
      <td>11.446047</td>
      <td>208.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>1482</th>
      <td>1966.0</td>
      <td>Wilt Chamberlain</td>
      <td>C</td>
      <td>29.0</td>
      <td>79.0</td>
      <td>0.0</td>
      <td>3737.0</td>
      <td>0.272625</td>
      <td>0.547</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>18.717688</td>
      <td>3.988226</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.647311</td>
      <td>25.518865</td>
      <td>216.0</td>
      <td>124.0</td>
    </tr>
    <tr>
      <th>1589</th>
      <td>1967.0</td>
      <td>Wilt Chamberlain</td>
      <td>C</td>
      <td>30.0</td>
      <td>81.0</td>
      <td>0.0</td>
      <td>3682.0</td>
      <td>0.259098</td>
      <td>0.637</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>19.134166</td>
      <td>6.159696</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.398153</td>
      <td>19.124389</td>
      <td>216.0</td>
      <td>124.0</td>
    </tr>
    <tr>
      <th>1711</th>
      <td>1968.0</td>
      <td>Wilt Chamberlain</td>
      <td>C</td>
      <td>31.0</td>
      <td>82.0</td>
      <td>0.0</td>
      <td>3836.0</td>
      <td>0.231804</td>
      <td>0.557</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>18.319082</td>
      <td>6.588113</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.501564</td>
      <td>18.694473</td>
      <td>216.0</td>
      <td>124.0</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>1969.0</td>
      <td>Wes Unseld</td>
      <td>C</td>
      <td>22.0</td>
      <td>82.0</td>
      <td>0.0</td>
      <td>2970.0</td>
      <td>0.219394</td>
      <td>0.515</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>18.072727</td>
      <td>2.581818</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.345455</td>
      <td>13.709091</td>
      <td>201.0</td>
      <td>111.0</td>
    </tr>
    <tr>
      <th>2148</th>
      <td>1970.0</td>
      <td>Willis Reed</td>
      <td>C</td>
      <td>27.0</td>
      <td>81.0</td>
      <td>0.0</td>
      <td>3089.0</td>
      <td>0.236581</td>
      <td>0.552</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>13.122693</td>
      <td>1.876335</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.344772</td>
      <td>20.453221</td>
      <td>206.0</td>
      <td>106.0</td>
    </tr>
    <tr>
      <th>2406</th>
      <td>1972.0</td>
      <td>Kareem Abdul-Jabbar</td>
      <td>C</td>
      <td>24.0</td>
      <td>81.0</td>
      <td>0.0</td>
      <td>3583.0</td>
      <td>0.300419</td>
      <td>0.603</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>13.523863</td>
      <td>3.717555</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.361150</td>
      <td>28.353893</td>
      <td>218.0</td>
      <td>102.0</td>
    </tr>
    <tr>
      <th>2665</th>
      <td>1973.0</td>
      <td>Dave Cowens</td>
      <td>C</td>
      <td>24.0</td>
      <td>82.0</td>
      <td>0.0</td>
      <td>3425.0</td>
      <td>0.190248</td>
      <td>0.481</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>13.969051</td>
      <td>3.500146</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.268905</td>
      <td>17.700438</td>
      <td>206.0</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>3195</th>
      <td>1975.0</td>
      <td>Bob McAdoo</td>
      <td>C</td>
      <td>23.0</td>
      <td>82.0</td>
      <td>0.0</td>
      <td>3539.0</td>
      <td>0.262447</td>
      <td>0.569</td>
      <td>0.000</td>
      <td>...</td>
      <td>8.626166</td>
      <td>11.749082</td>
      <td>1.820853</td>
      <td>0.935858</td>
      <td>1.769992</td>
      <td>0.000000</td>
      <td>2.827917</td>
      <td>28.797966</td>
      <td>206.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>3303</th>
      <td>1976.0</td>
      <td>Kareem Abdul-Jabbar</td>
      <td>C</td>
      <td>28.0</td>
      <td>82.0</td>
      <td>0.0</td>
      <td>3379.0</td>
      <td>0.289790</td>
      <td>0.567</td>
      <td>0.000</td>
      <td>...</td>
      <td>11.836638</td>
      <td>14.734537</td>
      <td>4.400118</td>
      <td>1.267831</td>
      <td>3.601065</td>
      <td>0.000000</td>
      <td>3.110980</td>
      <td>24.237940</td>
      <td>218.0</td>
      <td>102.0</td>
    </tr>
    <tr>
      <th>3536</th>
      <td>1977.0</td>
      <td>Kareem Abdul-Jabbar</td>
      <td>C</td>
      <td>29.0</td>
      <td>82.0</td>
      <td>0.0</td>
      <td>3016.0</td>
      <td>0.331830</td>
      <td>0.608</td>
      <td>0.000</td>
      <td>...</td>
      <td>9.835544</td>
      <td>13.010610</td>
      <td>3.807692</td>
      <td>1.205570</td>
      <td>3.115385</td>
      <td>0.000000</td>
      <td>3.127321</td>
      <td>25.687003</td>
      <td>218.0</td>
      <td>102.0</td>
    </tr>
    <tr>
      <th>4131</th>
      <td>1978.0</td>
      <td>Bill Walton</td>
      <td>C</td>
      <td>25.0</td>
      <td>58.0</td>
      <td>0.0</td>
      <td>1929.0</td>
      <td>0.462830</td>
      <td>0.554</td>
      <td>0.000</td>
      <td>...</td>
      <td>12.093313</td>
      <td>14.295490</td>
      <td>5.430793</td>
      <td>1.119751</td>
      <td>2.724728</td>
      <td>3.844479</td>
      <td>2.706065</td>
      <td>20.472784</td>
      <td>211.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>4334</th>
      <td>1979.0</td>
      <td>Moses Malone</td>
      <td>C</td>
      <td>23.0</td>
      <td>82.0</td>
      <td>0.0</td>
      <td>3390.0</td>
      <td>0.251681</td>
      <td>0.604</td>
      <td>0.000</td>
      <td>...</td>
      <td>9.100885</td>
      <td>15.334513</td>
      <td>1.561062</td>
      <td>0.838938</td>
      <td>1.263717</td>
      <td>3.461947</td>
      <td>2.368142</td>
      <td>21.568142</td>
      <td>208.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>4464</th>
      <td>1980.0</td>
      <td>Kareem Abdul-Jabbar</td>
      <td>C</td>
      <td>32.0</td>
      <td>82.0</td>
      <td>0.0</td>
      <td>3143.0</td>
      <td>0.289787</td>
      <td>0.639</td>
      <td>0.001</td>
      <td>...</td>
      <td>7.972001</td>
      <td>10.148266</td>
      <td>4.249443</td>
      <td>0.927776</td>
      <td>3.207127</td>
      <td>3.401845</td>
      <td>2.474069</td>
      <td>23.297486</td>
      <td>218.0</td>
      <td>102.0</td>
    </tr>
    <tr>
      <th>4854</th>
      <td>1981.0</td>
      <td>Julius Erving</td>
      <td>SF</td>
      <td>30.0</td>
      <td>82.0</td>
      <td>0.0</td>
      <td>2874.0</td>
      <td>0.314405</td>
      <td>0.572</td>
      <td>0.012</td>
      <td>...</td>
      <td>5.173278</td>
      <td>8.229645</td>
      <td>4.559499</td>
      <td>2.167015</td>
      <td>1.841336</td>
      <td>3.331942</td>
      <td>2.918580</td>
      <td>25.227557</td>
      <td>201.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>5271</th>
      <td>1982.0</td>
      <td>Moses Malone</td>
      <td>C</td>
      <td>26.0</td>
      <td>81.0</td>
      <td>81.0</td>
      <td>3398.0</td>
      <td>0.283932</td>
      <td>0.576</td>
      <td>0.003</td>
      <td>...</td>
      <td>6.674514</td>
      <td>12.586227</td>
      <td>1.504414</td>
      <td>0.805180</td>
      <td>1.324308</td>
      <td>3.114773</td>
      <td>2.203649</td>
      <td>26.698058</td>
      <td>208.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>5595</th>
      <td>1983.0</td>
      <td>Moses Malone</td>
      <td>C</td>
      <td>27.0</td>
      <td>78.0</td>
      <td>78.0</td>
      <td>2922.0</td>
      <td>0.309240</td>
      <td>0.578</td>
      <td>0.001</td>
      <td>...</td>
      <td>9.227926</td>
      <td>14.710472</td>
      <td>1.244353</td>
      <td>1.096509</td>
      <td>1.934292</td>
      <td>3.252567</td>
      <td>2.537988</td>
      <td>23.507187</td>
      <td>208.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>5750</th>
      <td>1984.0</td>
      <td>Larry Bird</td>
      <td>PF</td>
      <td>27.0</td>
      <td>79.0</td>
      <td>77.0</td>
      <td>3028.0</td>
      <td>0.287715</td>
      <td>0.552</td>
      <td>0.047</td>
      <td>...</td>
      <td>7.311757</td>
      <td>9.463672</td>
      <td>6.182299</td>
      <td>1.712021</td>
      <td>0.820343</td>
      <td>2.817701</td>
      <td>2.342140</td>
      <td>22.684280</td>
      <td>206.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>6042</th>
      <td>1985.0</td>
      <td>Larry Bird</td>
      <td>SF</td>
      <td>28.0</td>
      <td>80.0</td>
      <td>77.0</td>
      <td>3161.0</td>
      <td>0.301803</td>
      <td>0.585</td>
      <td>0.074</td>
      <td>...</td>
      <td>7.721607</td>
      <td>9.589370</td>
      <td>6.047453</td>
      <td>1.469155</td>
      <td>1.116102</td>
      <td>2.824423</td>
      <td>2.368871</td>
      <td>26.137298</td>
      <td>206.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>6339</th>
      <td>1986.0</td>
      <td>Larry Bird</td>
      <td>SF</td>
      <td>29.0</td>
      <td>82.0</td>
      <td>81.0</td>
      <td>3113.0</td>
      <td>0.296049</td>
      <td>0.580</td>
      <td>0.121</td>
      <td>...</td>
      <td>7.112111</td>
      <td>9.309348</td>
      <td>6.441375</td>
      <td>1.919692</td>
      <td>0.589785</td>
      <td>3.076132</td>
      <td>2.104722</td>
      <td>24.458721</td>
      <td>206.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>6761</th>
      <td>1987.0</td>
      <td>Magic Johnson</td>
      <td>PG</td>
      <td>27.0</td>
      <td>80.0</td>
      <td>80.0</td>
      <td>2904.0</td>
      <td>0.334711</td>
      <td>0.602</td>
      <td>0.030</td>
      <td>...</td>
      <td>4.735537</td>
      <td>6.247934</td>
      <td>12.111570</td>
      <td>1.710744</td>
      <td>0.446281</td>
      <td>3.719008</td>
      <td>2.082645</td>
      <td>23.665289</td>
      <td>201.0</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>7103</th>
      <td>1988.0</td>
      <td>Michael Jordan</td>
      <td>SG</td>
      <td>24.0</td>
      <td>82.0</td>
      <td>82.0</td>
      <td>3311.0</td>
      <td>0.344669</td>
      <td>0.603</td>
      <td>0.027</td>
      <td>...</td>
      <td>3.370583</td>
      <td>4.881909</td>
      <td>5.273331</td>
      <td>2.816068</td>
      <td>1.424343</td>
      <td>2.739958</td>
      <td>2.935669</td>
      <td>31.183328</td>
      <td>198.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>7435</th>
      <td>1989.0</td>
      <td>Magic Johnson</td>
      <td>PG</td>
      <td>29.0</td>
      <td>77.0</td>
      <td>77.0</td>
      <td>2886.0</td>
      <td>0.335551</td>
      <td>0.625</td>
      <td>0.165</td>
      <td>...</td>
      <td>6.187110</td>
      <td>7.571726</td>
      <td>12.324324</td>
      <td>1.721414</td>
      <td>0.274428</td>
      <td>3.891892</td>
      <td>2.145530</td>
      <td>21.580042</td>
      <td>201.0</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>7807</th>
      <td>1990.0</td>
      <td>Magic Johnson</td>
      <td>PG</td>
      <td>30.0</td>
      <td>79.0</td>
      <td>79.0</td>
      <td>2937.0</td>
      <td>0.326047</td>
      <td>0.622</td>
      <td>0.243</td>
      <td>...</td>
      <td>4.829418</td>
      <td>6.398366</td>
      <td>11.117467</td>
      <td>1.617978</td>
      <td>0.416752</td>
      <td>3.542390</td>
      <td>2.046987</td>
      <td>21.634321</td>
      <td>201.0</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>8181</th>
      <td>1991.0</td>
      <td>Michael Jordan</td>
      <td>SG</td>
      <td>27.0</td>
      <td>82.0</td>
      <td>82.0</td>
      <td>3034.0</td>
      <td>0.374951</td>
      <td>0.605</td>
      <td>0.051</td>
      <td>...</td>
      <td>4.437706</td>
      <td>5.837838</td>
      <td>5.375082</td>
      <td>2.646012</td>
      <td>0.984838</td>
      <td>2.396836</td>
      <td>2.717205</td>
      <td>30.613052</td>
      <td>198.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>8547</th>
      <td>1992.0</td>
      <td>Michael Jordan</td>
      <td>SG</td>
      <td>28.0</td>
      <td>80.0</td>
      <td>80.0</td>
      <td>3102.0</td>
      <td>0.321470</td>
      <td>0.579</td>
      <td>0.055</td>
      <td>...</td>
      <td>4.874275</td>
      <td>5.930368</td>
      <td>5.675048</td>
      <td>2.112186</td>
      <td>0.870406</td>
      <td>2.321083</td>
      <td>2.332689</td>
      <td>27.899420</td>
      <td>198.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>8754</th>
      <td>1993.0</td>
      <td>Charles Barkley</td>
      <td>PF</td>
      <td>29.0</td>
      <td>76.0</td>
      <td>76.0</td>
      <td>2859.0</td>
      <td>0.326128</td>
      <td>0.596</td>
      <td>0.160</td>
      <td>...</td>
      <td>8.700944</td>
      <td>11.685205</td>
      <td>4.847849</td>
      <td>1.498426</td>
      <td>0.931794</td>
      <td>2.933893</td>
      <td>2.467996</td>
      <td>24.478489</td>
      <td>198.0</td>
      <td>114.0</td>
    </tr>
    <tr>
      <th>9333</th>
      <td>1994.0</td>
      <td>Hakeem Olajuwon</td>
      <td>C</td>
      <td>31.0</td>
      <td>80.0</td>
      <td>80.0</td>
      <td>3277.0</td>
      <td>0.277937</td>
      <td>0.565</td>
      <td>0.011</td>
      <td>...</td>
      <td>7.975587</td>
      <td>10.491303</td>
      <td>3.152884</td>
      <td>1.406164</td>
      <td>3.262740</td>
      <td>2.977113</td>
      <td>3.174855</td>
      <td>23.992676</td>
      <td>213.0</td>
      <td>115.0</td>
    </tr>
    <tr>
      <th>9738</th>
      <td>1995.0</td>
      <td>David Robinson</td>
      <td>C</td>
      <td>29.0</td>
      <td>81.0</td>
      <td>81.0</td>
      <td>3074.0</td>
      <td>0.340794</td>
      <td>0.602</td>
      <td>0.013</td>
      <td>...</td>
      <td>7.530254</td>
      <td>10.270657</td>
      <td>2.763826</td>
      <td>1.569291</td>
      <td>3.068315</td>
      <td>2.728692</td>
      <td>2.693559</td>
      <td>26.209499</td>
      <td>216.0</td>
      <td>106.0</td>
    </tr>
    <tr>
      <th>10020</th>
      <td>1996.0</td>
      <td>Michael Jordan</td>
      <td>SG</td>
      <td>32.0</td>
      <td>82.0</td>
      <td>82.0</td>
      <td>3090.0</td>
      <td>0.342524</td>
      <td>0.582</td>
      <td>0.141</td>
      <td>...</td>
      <td>4.601942</td>
      <td>6.326214</td>
      <td>4.100971</td>
      <td>2.097087</td>
      <td>0.489320</td>
      <td>2.295146</td>
      <td>2.271845</td>
      <td>29.021359</td>
      <td>198.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>10499</th>
      <td>1997.0</td>
      <td>Karl Malone</td>
      <td>PF</td>
      <td>33.0</td>
      <td>82.0</td>
      <td>82.0</td>
      <td>2998.0</td>
      <td>0.347031</td>
      <td>0.600</td>
      <td>0.008</td>
      <td>...</td>
      <td>7.396931</td>
      <td>9.714476</td>
      <td>4.418946</td>
      <td>1.356905</td>
      <td>0.576384</td>
      <td>2.797865</td>
      <td>2.605737</td>
      <td>27.006004</td>
      <td>206.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>10910</th>
      <td>1998.0</td>
      <td>Michael Jordan</td>
      <td>SG</td>
      <td>34.0</td>
      <td>82.0</td>
      <td>82.0</td>
      <td>3181.0</td>
      <td>0.285193</td>
      <td>0.533</td>
      <td>0.067</td>
      <td>...</td>
      <td>3.904433</td>
      <td>5.375668</td>
      <td>3.202766</td>
      <td>1.595725</td>
      <td>0.509274</td>
      <td>2.093681</td>
      <td>1.708897</td>
      <td>26.674631</td>
      <td>198.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>11352</th>
      <td>1999.0</td>
      <td>Karl Malone</td>
      <td>PF</td>
      <td>35.0</td>
      <td>49.0</td>
      <td>49.0</td>
      <td>1832.0</td>
      <td>0.503057</td>
      <td>0.577</td>
      <td>0.001</td>
      <td>...</td>
      <td>6.995633</td>
      <td>9.098253</td>
      <td>3.949782</td>
      <td>1.218341</td>
      <td>0.550218</td>
      <td>3.183406</td>
      <td>2.633188</td>
      <td>22.873362</td>
      <td>206.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>11793</th>
      <td>2000.0</td>
      <td>Shaquille O'Neal</td>
      <td>C</td>
      <td>27.0</td>
      <td>79.0</td>
      <td>79.0</td>
      <td>3163.0</td>
      <td>0.348277</td>
      <td>0.578</td>
      <td>0.001</td>
      <td>...</td>
      <td>8.445147</td>
      <td>12.269365</td>
      <td>3.403098</td>
      <td>0.409738</td>
      <td>2.720202</td>
      <td>2.538097</td>
      <td>2.902308</td>
      <td>26.678470</td>
      <td>216.0</td>
      <td>147.0</td>
    </tr>
    <tr>
      <th>12093</th>
      <td>2001.0</td>
      <td>Allen Iverson</td>
      <td>SG</td>
      <td>25.0</td>
      <td>71.0</td>
      <td>71.0</td>
      <td>2979.0</td>
      <td>0.290030</td>
      <td>0.518</td>
      <td>0.169</td>
      <td>...</td>
      <td>2.694864</td>
      <td>3.299094</td>
      <td>3.927492</td>
      <td>2.151057</td>
      <td>0.241692</td>
      <td>2.864048</td>
      <td>1.776435</td>
      <td>26.670695</td>
      <td>183.0</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>12459</th>
      <td>2002.0</td>
      <td>Tim Duncan</td>
      <td>PF</td>
      <td>25.0</td>
      <td>82.0</td>
      <td>82.0</td>
      <td>3329.0</td>
      <td>0.291980</td>
      <td>0.576</td>
      <td>0.007</td>
      <td>...</td>
      <td>8.370081</td>
      <td>11.268249</td>
      <td>3.319916</td>
      <td>0.659658</td>
      <td>2.195254</td>
      <td>2.844097</td>
      <td>2.346651</td>
      <td>22.590568</td>
      <td>211.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>12849</th>
      <td>2003.0</td>
      <td>Tim Duncan</td>
      <td>PF</td>
      <td>26.0</td>
      <td>81.0</td>
      <td>81.0</td>
      <td>3181.0</td>
      <td>0.304433</td>
      <td>0.564</td>
      <td>0.016</td>
      <td>...</td>
      <td>8.872682</td>
      <td>11.803835</td>
      <td>3.576234</td>
      <td>0.622446</td>
      <td>2.682175</td>
      <td>2.806665</td>
      <td>2.614272</td>
      <td>21.321597</td>
      <td>211.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>13287</th>
      <td>2004.0</td>
      <td>Kevin Garnett</td>
      <td>PF</td>
      <td>27.0</td>
      <td>82.0</td>
      <td>82.0</td>
      <td>3231.0</td>
      <td>0.327577</td>
      <td>0.547</td>
      <td>0.027</td>
      <td>...</td>
      <td>9.961003</td>
      <td>12.690808</td>
      <td>4.557103</td>
      <td>1.337047</td>
      <td>1.983287</td>
      <td>2.362117</td>
      <td>2.250696</td>
      <td>22.139276</td>
      <td>211.0</td>
      <td>108.0</td>
    </tr>
    <tr>
      <th>13921</th>
      <td>2005.0</td>
      <td>Steve Nash</td>
      <td>PG</td>
      <td>30.0</td>
      <td>75.0</td>
      <td>75.0</td>
      <td>2573.0</td>
      <td>0.307812</td>
      <td>0.606</td>
      <td>0.254</td>
      <td>...</td>
      <td>2.686358</td>
      <td>3.483871</td>
      <td>12.046638</td>
      <td>1.035367</td>
      <td>0.083949</td>
      <td>3.427905</td>
      <td>1.902837</td>
      <td>16.300039</td>
      <td>190.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>14378</th>
      <td>2006.0</td>
      <td>Steve Nash</td>
      <td>PG</td>
      <td>31.0</td>
      <td>79.0</td>
      <td>79.0</td>
      <td>2796.0</td>
      <td>0.300000</td>
      <td>0.632</td>
      <td>0.324</td>
      <td>...</td>
      <td>3.682403</td>
      <td>4.287554</td>
      <td>10.635193</td>
      <td>0.785408</td>
      <td>0.154506</td>
      <td>3.553648</td>
      <td>1.545064</td>
      <td>19.171674</td>
      <td>190.0</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>14825</th>
      <td>2007.0</td>
      <td>Dirk Nowitzki</td>
      <td>PF</td>
      <td>28.0</td>
      <td>78.0</td>
      <td>78.0</td>
      <td>2820.0</td>
      <td>0.352340</td>
      <td>0.605</td>
      <td>0.129</td>
      <td>...</td>
      <td>7.289362</td>
      <td>8.846809</td>
      <td>3.357447</td>
      <td>0.663830</td>
      <td>0.791489</td>
      <td>2.131915</td>
      <td>2.182979</td>
      <td>24.459574</td>
      <td>213.0</td>
      <td>111.0</td>
    </tr>
    <tr>
      <th>15027</th>
      <td>2008.0</td>
      <td>Kobe Bryant</td>
      <td>SG</td>
      <td>29.0</td>
      <td>82.0</td>
      <td>82.0</td>
      <td>3192.0</td>
      <td>0.272932</td>
      <td>0.576</td>
      <td>0.246</td>
      <td>...</td>
      <td>4.770677</td>
      <td>5.830827</td>
      <td>4.973684</td>
      <td>1.703008</td>
      <td>0.451128</td>
      <td>2.898496</td>
      <td>2.560150</td>
      <td>26.199248</td>
      <td>198.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>15642</th>
      <td>2009.0</td>
      <td>LeBron James</td>
      <td>SF</td>
      <td>24.0</td>
      <td>81.0</td>
      <td>81.0</td>
      <td>3054.0</td>
      <td>0.373674</td>
      <td>0.591</td>
      <td>0.238</td>
      <td>...</td>
      <td>5.976424</td>
      <td>7.225933</td>
      <td>6.919450</td>
      <td>1.614931</td>
      <td>1.096267</td>
      <td>2.840864</td>
      <td>1.638507</td>
      <td>27.159136</td>
      <td>203.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>16109</th>
      <td>2010.0</td>
      <td>LeBron James</td>
      <td>SF</td>
      <td>25.0</td>
      <td>76.0</td>
      <td>76.0</td>
      <td>2966.0</td>
      <td>0.377478</td>
      <td>0.604</td>
      <td>0.253</td>
      <td>...</td>
      <td>5.862441</td>
      <td>6.724208</td>
      <td>7.901551</td>
      <td>1.517195</td>
      <td>0.934592</td>
      <td>3.167903</td>
      <td>1.444370</td>
      <td>27.406608</td>
      <td>203.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>16745</th>
      <td>2011.0</td>
      <td>Derrick Rose</td>
      <td>PG</td>
      <td>22.0</td>
      <td>81.0</td>
      <td>81.0</td>
      <td>3026.0</td>
      <td>0.279577</td>
      <td>0.550</td>
      <td>0.241</td>
      <td>...</td>
      <td>2.962327</td>
      <td>3.925975</td>
      <td>7.411765</td>
      <td>1.011236</td>
      <td>0.606742</td>
      <td>3.307336</td>
      <td>1.617978</td>
      <td>24.103106</td>
      <td>190.0</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>17060</th>
      <td>2012.0</td>
      <td>LeBron James</td>
      <td>SF</td>
      <td>27.0</td>
      <td>62.0</td>
      <td>62.0</td>
      <td>2326.0</td>
      <td>0.475150</td>
      <td>0.605</td>
      <td>0.127</td>
      <td>...</td>
      <td>6.159931</td>
      <td>7.614789</td>
      <td>5.989682</td>
      <td>1.779880</td>
      <td>0.773861</td>
      <td>3.296647</td>
      <td>1.485813</td>
      <td>26.048151</td>
      <td>203.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>17499</th>
      <td>2013.0</td>
      <td>LeBron James</td>
      <td>PF</td>
      <td>28.0</td>
      <td>76.0</td>
      <td>76.0</td>
      <td>2877.0</td>
      <td>0.395412</td>
      <td>0.640</td>
      <td>0.188</td>
      <td>...</td>
      <td>6.419187</td>
      <td>7.632951</td>
      <td>6.894682</td>
      <td>1.614181</td>
      <td>0.838373</td>
      <td>2.827946</td>
      <td>1.376434</td>
      <td>25.476538</td>
      <td>203.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>17897</th>
      <td>2014.0</td>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>25.0</td>
      <td>81.0</td>
      <td>81.0</td>
      <td>3122.0</td>
      <td>0.343626</td>
      <td>0.635</td>
      <td>0.291</td>
      <td>...</td>
      <td>6.226778</td>
      <td>6.895580</td>
      <td>5.131326</td>
      <td>1.187700</td>
      <td>0.680333</td>
      <td>3.286355</td>
      <td>2.006406</td>
      <td>29.900064</td>
      <td>206.0</td>
      <td>108.0</td>
    </tr>
    <tr>
      <th>18344</th>
      <td>2015.0</td>
      <td>Stephen Curry</td>
      <td>PG</td>
      <td>26.0</td>
      <td>80.0</td>
      <td>80.0</td>
      <td>2613.0</td>
      <td>0.385763</td>
      <td>0.638</td>
      <td>0.482</td>
      <td>...</td>
      <td>3.926521</td>
      <td>4.698048</td>
      <td>8.528129</td>
      <td>2.245695</td>
      <td>0.220436</td>
      <td>3.430540</td>
      <td>2.176808</td>
      <td>26.176808</td>
      <td>190.0</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>18842</th>
      <td>2016.0</td>
      <td>Stephen Curry</td>
      <td>PG</td>
      <td>27.0</td>
      <td>79.0</td>
      <td>79.0</td>
      <td>2700.0</td>
      <td>0.420000</td>
      <td>0.669</td>
      <td>0.554</td>
      <td>...</td>
      <td>4.826667</td>
      <td>5.733333</td>
      <td>7.026667</td>
      <td>2.253333</td>
      <td>0.200000</td>
      <td>3.493333</td>
      <td>2.146667</td>
      <td>31.666667</td>
      <td>190.0</td>
      <td>86.0</td>
    </tr>
  </tbody>
</table>
<p>59 rows × 51 columns</p>
</div>




```python
mvp_pred = pd.DataFrame(index=mvp_stats.loc[:, 'Player'].values, data={'Real': mvp_stats.loc[:, 'Pos'].values,
    'Predicted':encoder.inverse_transform(model.predict(Xnorm[mvp_stats.index, :]))})
```


```python
mvp_pred
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted</th>
      <th>Real</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bob Pettit</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Bob Cousy</th>
      <td>PG</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>Bill Russell</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Bob Pettit</th>
      <td>C</td>
      <td>PF</td>
    </tr>
    <tr>
      <th>Wilt Chamberlain</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Bill Russell</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Bill Russell</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Bill Russell</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Oscar Robertson</th>
      <td>SG</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>Bill Russell</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Wilt Chamberlain</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Wilt Chamberlain</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Wilt Chamberlain</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Wes Unseld</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Willis Reed</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Kareem Abdul-Jabbar</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Dave Cowens</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Bob McAdoo</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Kareem Abdul-Jabbar</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Kareem Abdul-Jabbar</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Bill Walton</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Moses Malone</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Kareem Abdul-Jabbar</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Julius Erving</th>
      <td>SF</td>
      <td>SF</td>
    </tr>
    <tr>
      <th>Moses Malone</th>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Moses Malone</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Larry Bird</th>
      <td>PF</td>
      <td>PF</td>
    </tr>
    <tr>
      <th>Larry Bird</th>
      <td>PF</td>
      <td>SF</td>
    </tr>
    <tr>
      <th>Larry Bird</th>
      <td>SF</td>
      <td>SF</td>
    </tr>
    <tr>
      <th>Magic Johnson</th>
      <td>PG</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>Michael Jordan</th>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>Magic Johnson</th>
      <td>PG</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>Magic Johnson</th>
      <td>PG</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>Michael Jordan</th>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>Michael Jordan</th>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>Charles Barkley</th>
      <td>PF</td>
      <td>PF</td>
    </tr>
    <tr>
      <th>Hakeem Olajuwon</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>David Robinson</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Michael Jordan</th>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>Karl Malone</th>
      <td>PF</td>
      <td>PF</td>
    </tr>
    <tr>
      <th>Michael Jordan</th>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>Karl Malone</th>
      <td>PF</td>
      <td>PF</td>
    </tr>
    <tr>
      <th>Shaquille O'Neal</th>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>Allen Iverson</th>
      <td>PG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>Tim Duncan</th>
      <td>C</td>
      <td>PF</td>
    </tr>
    <tr>
      <th>Tim Duncan</th>
      <td>C</td>
      <td>PF</td>
    </tr>
    <tr>
      <th>Kevin Garnett</th>
      <td>C</td>
      <td>PF</td>
    </tr>
    <tr>
      <th>Steve Nash</th>
      <td>PG</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>Steve Nash</th>
      <td>PG</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>Dirk Nowitzki</th>
      <td>C</td>
      <td>PF</td>
    </tr>
    <tr>
      <th>Kobe Bryant</th>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>LeBron James</th>
      <td>SF</td>
      <td>SF</td>
    </tr>
    <tr>
      <th>LeBron James</th>
      <td>SG</td>
      <td>SF</td>
    </tr>
    <tr>
      <th>Derrick Rose</th>
      <td>PG</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>LeBron James</th>
      <td>SF</td>
      <td>SF</td>
    </tr>
    <tr>
      <th>LeBron James</th>
      <td>SF</td>
      <td>PF</td>
    </tr>
    <tr>
      <th>Kevin Durant</th>
      <td>SF</td>
      <td>SF</td>
    </tr>
    <tr>
      <th>Stephen Curry</th>
      <td>PG</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>Stephen Curry</th>
      <td>PG</td>
      <td>PG</td>
    </tr>
  </tbody>
</table>
</div>



The model gets right most of the players, and the errors are always for a contiguous position (it is interesting that the model does it without having been provided with any information about the distances between the labels.)

# Does year metter? #

The definition of what a forward, or a center are always changing (in the very recent years, there is, for example, a trend towards having scoring point guards (as Stephen Curry) and forwards that direct the game instead of the guard (as Lebron James). 
Also, the physical requirements are increasing, and a height that in the 50's could characterize you as a center you will be a forward today.

We will follow the first and last MVP's, Stephen Curry and Bob Pettit, and see where our model will put them in different years in the NBA history.


```python
curry2017 = data[(data['Player'] == 'Stephen Curry') & (data['Year']==2017)] 
pettit1956 = data[(data['Player'] == 'Bob Pettit') & (data['Year']==1956)]
```


```python
time_travel_curry = pd.concat([curry2017 for year in range(1956, 2018)], axis=0)
time_travel_curry['Year'] = range(1956, 2018)

X = time_travel_curry.drop(['Player', 'Pos', 'G', 'GS', 'MP'], axis=1).as_matrix()
y = time_travel_curry['Pos'].as_matrix()

y_cat = encoder.transform(y)
Xnorm = scaler.transform(X)

time_travel_curry_pred = pd.DataFrame(index=time_travel_curry.loc[:, 'Year'].values, 
                                data={'Real': time_travel_curry.loc[:, 'Pos'].values,
    'Predicted':encoder.inverse_transform(model.predict(Xnorm))})


time_travel_pettit = pd.concat([pettit1956 for year in range(1956, 2018)], axis=0)
time_travel_pettit['Year'] = range(1956, 2018)

X = time_travel_pettit.drop(['Player', 'Pos', 'G', 'GS', 'MP'], axis=1).as_matrix()
y = time_travel_pettit['Pos'].as_matrix()

y_cat = encoder.transform(y)
Xnorm = scaler.transform(X)

time_travel_pettit_pred = pd.DataFrame(index=time_travel_pettit.loc[:, 'Year'].values, 
                                data={'Real': time_travel_pettit.loc[:, 'Pos'].values,
    'Predicted':encoder.inverse_transform(model.predict(Xnorm))})
```


```python
pd.concat([time_travel_curry_pred,time_travel_pettit_pred],axis=1,keys=['Stephen Curry','Bob Pettit'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Stephen Curry</th>
      <th colspan="2" halign="left">Bob Pettit</th>
    </tr>
    <tr>
      <th></th>
      <th>Predicted</th>
      <th>Real</th>
      <th>Predicted</th>
      <th>Real</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1956</th>
      <td>SG</td>
      <td>PG</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>SG</td>
      <td>PG</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>SG</td>
      <td>PG</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1959</th>
      <td>SG</td>
      <td>PG</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1960</th>
      <td>SG</td>
      <td>PG</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>SG</td>
      <td>PG</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>SG</td>
      <td>PG</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>SG</td>
      <td>PG</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>SG</td>
      <td>PG</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>SG</td>
      <td>PG</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>SG</td>
      <td>PG</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>SG</td>
      <td>PG</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>SG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1969</th>
      <td>SG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>SG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>SG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>SG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1978</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1979</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1981</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1985</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>PG</td>
      <td>PG</td>
      <td>PF</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
<p>62 rows × 4 columns</p>
</div>



Curry is labeled as a point guards (his real position) from 1973 until today, and as a shooting guard before that. Perhaps because of his heigh (191cm), or perhaps because he is too much of a scorer. Bob Pettit is labeled as a center until 1967, and as a power forward after that (he played both roles, but nowadays he would have difficulties to play as a center, and would be for sure a forwards, perhaps even a small forward). 

# Changing positions #

Many players go towards more interior roles with age, as they lose velocity. We will follow two cases, Magic Johnson, and Michael Jordan. Both of the retire, and return years later with more interior roles.


```python
magic = data[(data['Player'] == 'Magic Johnson')] 
jordan = data[(data['Player'] == 'Michael Jordan')]
```


```python
# Magic
X = magic.drop(['Player', 'Pos', 'G', 'GS', 'MP'], axis=1).as_matrix()
y = magic['Pos'].as_matrix()

y_cat = encoder.transform(y)
Xnorm = scaler.transform(X)

magic_pred = pd.DataFrame(index=magic.loc[:, 'Age'].values, 
                                data={'Real': magic.loc[:, 'Pos'].values,
    'Predicted':encoder.inverse_transform(model.predict(Xnorm))})

# Jordan
X = jordan.drop(['Player', 'Pos', 'G', 'GS', 'MP'], axis=1).as_matrix()
y = jordan['Pos'].as_matrix()

y_cat = encoder.transform(y)
Xnorm = scaler.transform(X)

jordan_pred = pd.DataFrame(index=jordan.loc[:, 'Age'].values, 
                                data={'Real': jordan.loc[:, 'Pos'].values,
    'Predicted':encoder.inverse_transform(model.predict(Xnorm))})
```


```python
pd.concat([magic_pred,jordan_pred],axis=1,keys=['Magic Johnson','Michael Jordan'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Magic Johnson</th>
      <th colspan="2" halign="left">Michael Jordan</th>
    </tr>
    <tr>
      <th></th>
      <th>Predicted</th>
      <th>Real</th>
      <th>Predicted</th>
      <th>Real</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20.0</th>
      <td>SF</td>
      <td>SG</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21.0</th>
      <td>SG</td>
      <td>SG</td>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>22.0</th>
      <td>SG</td>
      <td>SG</td>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>23.0</th>
      <td>PG</td>
      <td>SG</td>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>24.0</th>
      <td>PG</td>
      <td>PG</td>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>25.0</th>
      <td>PG</td>
      <td>PG</td>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>26.0</th>
      <td>PG</td>
      <td>PG</td>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>27.0</th>
      <td>PG</td>
      <td>PG</td>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>28.0</th>
      <td>PG</td>
      <td>PG</td>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>29.0</th>
      <td>PG</td>
      <td>PG</td>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>30.0</th>
      <td>PG</td>
      <td>PG</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31.0</th>
      <td>PG</td>
      <td>PG</td>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>32.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>33.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>34.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>SG</td>
      <td>SG</td>
    </tr>
    <tr>
      <th>36.0</th>
      <td>SG</td>
      <td>PF</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>38.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>SF</td>
      <td>SF</td>
    </tr>
    <tr>
      <th>39.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>SF</td>
      <td>SF</td>
    </tr>
  </tbody>
</table>
</div>



The model is able to detect the conversion of Jordan into a forward at the end of his career, but not the return of Magic as a power forward. Also, in his rookie season, he is classified as a small forward instead of as a shooting guard (Magic was clearly and outlier in the data, a 205cm point guard who could easily play in the five position. It is even surprised that is properly labelled as a point guard during most of his career)

# How important are height and weight? #

A concern we have before training the model was that the model would use the height and weight as the main classifiers, and that would labelled incorrectly players as Magic Johnson (a 205 cm point guard), or Charles Barkley (a 196cm power forward). Almost surprisingly, it works properly on this two players.

We will use again the 2017 First NBA Team and play with the heights and weights of the players. Keeping constant all other statistics, we will change the height and weight and observe how the predicted positions change.


```python
first_team_stats
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Player</th>
      <th>Pos</th>
      <th>Age</th>
      <th>G</th>
      <th>GS</th>
      <th>MP</th>
      <th>PER</th>
      <th>TS%</th>
      <th>3PAr</th>
      <th>...</th>
      <th>DRB</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>TOV</th>
      <th>PF</th>
      <th>PTS</th>
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19326</th>
      <td>2017.0</td>
      <td>Anthony Davis</td>
      <td>C</td>
      <td>23.0</td>
      <td>75.0</td>
      <td>75.0</td>
      <td>2708.0</td>
      <td>0.365583</td>
      <td>0.579</td>
      <td>0.088</td>
      <td>...</td>
      <td>9.465288</td>
      <td>11.778434</td>
      <td>2.087149</td>
      <td>1.249631</td>
      <td>2.220089</td>
      <td>2.406204</td>
      <td>2.233383</td>
      <td>27.903988</td>
      <td>206.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>19399</th>
      <td>2017.0</td>
      <td>James Harden</td>
      <td>PG</td>
      <td>27.0</td>
      <td>81.0</td>
      <td>81.0</td>
      <td>2947.0</td>
      <td>0.333492</td>
      <td>0.613</td>
      <td>0.493</td>
      <td>...</td>
      <td>6.889718</td>
      <td>8.050221</td>
      <td>11.067526</td>
      <td>1.465898</td>
      <td>0.451985</td>
      <td>5.668137</td>
      <td>2.626400</td>
      <td>28.780455</td>
      <td>196.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>19444</th>
      <td>2017.0</td>
      <td>LeBron James</td>
      <td>SF</td>
      <td>32.0</td>
      <td>74.0</td>
      <td>74.0</td>
      <td>2794.0</td>
      <td>0.347888</td>
      <td>0.619</td>
      <td>0.254</td>
      <td>...</td>
      <td>6.996421</td>
      <td>8.246242</td>
      <td>8.323550</td>
      <td>1.185397</td>
      <td>0.566929</td>
      <td>3.904080</td>
      <td>1.726557</td>
      <td>25.176807</td>
      <td>203.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>19486</th>
      <td>2017.0</td>
      <td>Kawhi Leonard</td>
      <td>SF</td>
      <td>25.0</td>
      <td>74.0</td>
      <td>74.0</td>
      <td>2474.0</td>
      <td>0.400162</td>
      <td>0.611</td>
      <td>0.294</td>
      <td>...</td>
      <td>5.092967</td>
      <td>6.257074</td>
      <td>3.783347</td>
      <td>1.920776</td>
      <td>0.800323</td>
      <td>2.240905</td>
      <td>1.775263</td>
      <td>27.472918</td>
      <td>201.0</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>19671</th>
      <td>2017.0</td>
      <td>Russell Westbrook</td>
      <td>PG</td>
      <td>28.0</td>
      <td>81.0</td>
      <td>81.0</td>
      <td>2802.0</td>
      <td>0.393148</td>
      <td>0.554</td>
      <td>0.300</td>
      <td>...</td>
      <td>9.340471</td>
      <td>11.100642</td>
      <td>10.792291</td>
      <td>1.708779</td>
      <td>0.398287</td>
      <td>5.627409</td>
      <td>2.441113</td>
      <td>32.865096</td>
      <td>190.0</td>
      <td>90.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 51 columns</p>
</div>




```python
multiplier = np.arange(0.8,1.2,0.02)
growing_predicted = []

for p in first_team_stats.iterrows():
    growing = pd.concat([p[1].to_frame().T for x in multiplier], axis=0)
    growing['height'] = growing['height'] * multiplier
    growing['weight'] = growing['weight'] * (multiplier ** 3)

    X = growing.drop(['Player', 'Pos', 'G', 'GS', 'MP'], axis=1).as_matrix()
    y = growing['Pos'].as_matrix()

    y_cat = encoder.transform(y)
    Xnorm = scaler.transform(X)

    growing_predicted.append(pd.DataFrame(index=multiplier, data={'height': growing.loc[:, 'height'].values,
            'Real': growing.loc[:, 'Pos'].values, 'Predicted':encoder.inverse_transform(model.predict(Xnorm))}))
```

    /usr/local/lib/python3.6/site-packages/sklearn/utils/validation.py:429: DataConversionWarning: Data with input dtype object was converted to float64 by StandardScaler.
      warnings.warn(msg, _DataConversionWarning)



```python
pd.concat(growing_predicted,axis=1,keys=first_team_stats['Player'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Player</th>
      <th colspan="3" halign="left">Anthony Davis</th>
      <th colspan="3" halign="left">James Harden</th>
      <th colspan="3" halign="left">LeBron James</th>
      <th colspan="3" halign="left">Kawhi Leonard</th>
      <th colspan="3" halign="left">Russell Westbrook</th>
    </tr>
    <tr>
      <th></th>
      <th>Predicted</th>
      <th>Real</th>
      <th>height</th>
      <th>Predicted</th>
      <th>Real</th>
      <th>height</th>
      <th>Predicted</th>
      <th>Real</th>
      <th>height</th>
      <th>Predicted</th>
      <th>Real</th>
      <th>height</th>
      <th>Predicted</th>
      <th>Real</th>
      <th>height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.80</th>
      <td>PF</td>
      <td>C</td>
      <td>164.8</td>
      <td>PG</td>
      <td>PG</td>
      <td>156.8</td>
      <td>PG</td>
      <td>SF</td>
      <td>162.4</td>
      <td>PG</td>
      <td>SF</td>
      <td>160.8</td>
      <td>PG</td>
      <td>PG</td>
      <td>152</td>
    </tr>
    <tr>
      <th>0.82</th>
      <td>PF</td>
      <td>C</td>
      <td>168.92</td>
      <td>PG</td>
      <td>PG</td>
      <td>160.72</td>
      <td>PG</td>
      <td>SF</td>
      <td>166.46</td>
      <td>PG</td>
      <td>SF</td>
      <td>164.82</td>
      <td>PG</td>
      <td>PG</td>
      <td>155.8</td>
    </tr>
    <tr>
      <th>0.84</th>
      <td>PF</td>
      <td>C</td>
      <td>173.04</td>
      <td>PG</td>
      <td>PG</td>
      <td>164.64</td>
      <td>PG</td>
      <td>SF</td>
      <td>170.52</td>
      <td>PG</td>
      <td>SF</td>
      <td>168.84</td>
      <td>PG</td>
      <td>PG</td>
      <td>159.6</td>
    </tr>
    <tr>
      <th>0.86</th>
      <td>PF</td>
      <td>C</td>
      <td>177.16</td>
      <td>PG</td>
      <td>PG</td>
      <td>168.56</td>
      <td>PG</td>
      <td>SF</td>
      <td>174.58</td>
      <td>PG</td>
      <td>SF</td>
      <td>172.86</td>
      <td>PG</td>
      <td>PG</td>
      <td>163.4</td>
    </tr>
    <tr>
      <th>0.88</th>
      <td>PF</td>
      <td>C</td>
      <td>181.28</td>
      <td>PG</td>
      <td>PG</td>
      <td>172.48</td>
      <td>PG</td>
      <td>SF</td>
      <td>178.64</td>
      <td>PG</td>
      <td>SF</td>
      <td>176.88</td>
      <td>PG</td>
      <td>PG</td>
      <td>167.2</td>
    </tr>
    <tr>
      <th>0.90</th>
      <td>PF</td>
      <td>C</td>
      <td>185.4</td>
      <td>PG</td>
      <td>PG</td>
      <td>176.4</td>
      <td>PG</td>
      <td>SF</td>
      <td>182.7</td>
      <td>SG</td>
      <td>SF</td>
      <td>180.9</td>
      <td>PG</td>
      <td>PG</td>
      <td>171</td>
    </tr>
    <tr>
      <th>0.92</th>
      <td>PF</td>
      <td>C</td>
      <td>189.52</td>
      <td>PG</td>
      <td>PG</td>
      <td>180.32</td>
      <td>PG</td>
      <td>SF</td>
      <td>186.76</td>
      <td>SG</td>
      <td>SF</td>
      <td>184.92</td>
      <td>PG</td>
      <td>PG</td>
      <td>174.8</td>
    </tr>
    <tr>
      <th>0.94</th>
      <td>PF</td>
      <td>C</td>
      <td>193.64</td>
      <td>PG</td>
      <td>PG</td>
      <td>184.24</td>
      <td>PG</td>
      <td>SF</td>
      <td>190.82</td>
      <td>SG</td>
      <td>SF</td>
      <td>188.94</td>
      <td>PG</td>
      <td>PG</td>
      <td>178.6</td>
    </tr>
    <tr>
      <th>0.96</th>
      <td>PF</td>
      <td>C</td>
      <td>197.76</td>
      <td>PG</td>
      <td>PG</td>
      <td>188.16</td>
      <td>SF</td>
      <td>SF</td>
      <td>194.88</td>
      <td>SG</td>
      <td>SF</td>
      <td>192.96</td>
      <td>PG</td>
      <td>PG</td>
      <td>182.4</td>
    </tr>
    <tr>
      <th>0.98</th>
      <td>PF</td>
      <td>C</td>
      <td>201.88</td>
      <td>PG</td>
      <td>PG</td>
      <td>192.08</td>
      <td>SF</td>
      <td>SF</td>
      <td>198.94</td>
      <td>SG</td>
      <td>SF</td>
      <td>196.98</td>
      <td>PG</td>
      <td>PG</td>
      <td>186.2</td>
    </tr>
    <tr>
      <th>1.00</th>
      <td>PF</td>
      <td>C</td>
      <td>206</td>
      <td>PG</td>
      <td>PG</td>
      <td>196</td>
      <td>PF</td>
      <td>SF</td>
      <td>203</td>
      <td>SF</td>
      <td>SF</td>
      <td>201</td>
      <td>PG</td>
      <td>PG</td>
      <td>190</td>
    </tr>
    <tr>
      <th>1.02</th>
      <td>PF</td>
      <td>C</td>
      <td>210.12</td>
      <td>PG</td>
      <td>PG</td>
      <td>199.92</td>
      <td>PF</td>
      <td>SF</td>
      <td>207.06</td>
      <td>SF</td>
      <td>SF</td>
      <td>205.02</td>
      <td>PG</td>
      <td>PG</td>
      <td>193.8</td>
    </tr>
    <tr>
      <th>1.04</th>
      <td>C</td>
      <td>C</td>
      <td>214.24</td>
      <td>PG</td>
      <td>PG</td>
      <td>203.84</td>
      <td>C</td>
      <td>SF</td>
      <td>211.12</td>
      <td>SF</td>
      <td>SF</td>
      <td>209.04</td>
      <td>SF</td>
      <td>PG</td>
      <td>197.6</td>
    </tr>
    <tr>
      <th>1.06</th>
      <td>C</td>
      <td>C</td>
      <td>218.36</td>
      <td>PG</td>
      <td>PG</td>
      <td>207.76</td>
      <td>C</td>
      <td>SF</td>
      <td>215.18</td>
      <td>PF</td>
      <td>SF</td>
      <td>213.06</td>
      <td>PF</td>
      <td>PG</td>
      <td>201.4</td>
    </tr>
    <tr>
      <th>1.08</th>
      <td>C</td>
      <td>C</td>
      <td>222.48</td>
      <td>PF</td>
      <td>PG</td>
      <td>211.68</td>
      <td>C</td>
      <td>SF</td>
      <td>219.24</td>
      <td>PF</td>
      <td>SF</td>
      <td>217.08</td>
      <td>PF</td>
      <td>PG</td>
      <td>205.2</td>
    </tr>
    <tr>
      <th>1.10</th>
      <td>C</td>
      <td>C</td>
      <td>226.6</td>
      <td>C</td>
      <td>PG</td>
      <td>215.6</td>
      <td>C</td>
      <td>SF</td>
      <td>223.3</td>
      <td>C</td>
      <td>SF</td>
      <td>221.1</td>
      <td>PF</td>
      <td>PG</td>
      <td>209</td>
    </tr>
    <tr>
      <th>1.12</th>
      <td>C</td>
      <td>C</td>
      <td>230.72</td>
      <td>C</td>
      <td>PG</td>
      <td>219.52</td>
      <td>C</td>
      <td>SF</td>
      <td>227.36</td>
      <td>C</td>
      <td>SF</td>
      <td>225.12</td>
      <td>PF</td>
      <td>PG</td>
      <td>212.8</td>
    </tr>
    <tr>
      <th>1.14</th>
      <td>C</td>
      <td>C</td>
      <td>234.84</td>
      <td>C</td>
      <td>PG</td>
      <td>223.44</td>
      <td>C</td>
      <td>SF</td>
      <td>231.42</td>
      <td>C</td>
      <td>SF</td>
      <td>229.14</td>
      <td>C</td>
      <td>PG</td>
      <td>216.6</td>
    </tr>
    <tr>
      <th>1.16</th>
      <td>C</td>
      <td>C</td>
      <td>238.96</td>
      <td>C</td>
      <td>PG</td>
      <td>227.36</td>
      <td>C</td>
      <td>SF</td>
      <td>235.48</td>
      <td>C</td>
      <td>SF</td>
      <td>233.16</td>
      <td>C</td>
      <td>PG</td>
      <td>220.4</td>
    </tr>
    <tr>
      <th>1.18</th>
      <td>C</td>
      <td>C</td>
      <td>243.08</td>
      <td>C</td>
      <td>PG</td>
      <td>231.28</td>
      <td>C</td>
      <td>SF</td>
      <td>239.54</td>
      <td>C</td>
      <td>SF</td>
      <td>237.18</td>
      <td>C</td>
      <td>PG</td>
      <td>224.2</td>
    </tr>
  </tbody>
</table>
</div>



As we can see height matters, but it's not enough. Any player can be classified as a center if he is tall enough (very tall: Kawhi Leonard would need to be 221cm tall to be considered a center), but being short it's not enough to be considered a guard: a 165cm Anthony Davis would be still considered a power forward.


```python

```
