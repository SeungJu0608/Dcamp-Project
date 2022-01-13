import json
import pickle
import sys
import traceback
from io import StringIO
import copy

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from operator import itemgetter
from itertools import combinations
from random import *

from sklearn.neural_network import MLPClassifier


''' Prepare Data '''
prefix = 'drive/MyDrive/Dcamp/'
data_path = os.path.join(prefix, 'data', 'total.csv')

input_files = [ data_path ]

raw_data = [ pd.read_csv(file, header=0) for file in input_files ]
raw = pd.concat(raw_data)
data = copy.deepcopy(raw) # len(data.columns) : 25

data.drop(columns = ['col1', 'col2', 'col4', 'col5'], inplace = True)


''' Data Preprocess & EDA '''
# Missing Value
data.dropna(subset = ["col3", "col6", "col7"], inplace=True)
data.reset_index(drop = True, inplace = True)

# Feature engineering 
# Make col6 into categorical data
data_clean = data.copy()
# col6 < 0 to multiply (-1)
clean_value = data[(data['col6'] < 0)].index
data_clean.loc[clean_value, "col6"] = data.loc[clean_value, "col6"]*(-1)

# col6 > 900 
clean_value = data[(data['col6'] > 900)].index
data_clean.loc[clean_value, "col6"] = data.loc[clean_value, "col6"] - 900

# 100 < col6 < 900 
clean_value = data[(data['col6'] > 100) & (data['col6'] < 900)].index
data_clean.loc[clean_value, "col6"] = data.loc[clean_value, "col6"] - 100

# col6 < 10 :
mean_val = round(data_clean['col6'].mean())

clean_value = data_clean[(data_clean['col6'] < 10)].index
data_clean.loc[clean_value, "col6"] = mean_val

# divide col6
under = data_clean[data_clean['col6'] < 30].index
thir_y = data_clean[( 30 <= data_clean['col6'] ) & (data_clean['col6'] < 36) ].index
thir_e = data_clean[( 36 <= data_clean['col6'] ) & (data_clean['col6'] < 40) ].index
four_y = data_clean[( 40 <= data_clean['col6'] ) & (data_clean['col6'] < 46) ].index
four_e = data_clean[( 46 <= data_clean['col6'] ) & (data_clean['col6'] < 50) ].index
five = data_clean[(50 <= data_clean['col6']) & (data_clean['col6'] < 60)].index
over = data_clean[ 60 <= data_clean['col6'] ].index
data_clean.loc[under, 'col6-2'] = "under"
data_clean.loc[thir_y, 'col6-2'] = "thir_y"
data_clean.loc[thir_e, 'col6-2'] = "thir_e"
data_clean.loc[four_y, 'col6-2'] = "four_y"
data_clean.loc[four_e, 'col6-2'] = "four_e"
data_clean.loc[five, 'col6-2'] = "five"
data_clean.loc[over, 'col6-2'] = "over"


# Make multi-label
## class : 
##     cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10, cat11, cat12
# Divide target class
data_clean["cat1"] = data_clean.apply(lambda x : 1 if ("cat1" in x["col3"].split(" ")) else (0), axis = 1)
data_clean["cat2"] = data_clean.apply(lambda x : 1 if ("cat2" in x["col3"].split(" ")) else(0), axis = 1)
data_clean["cat3"] = data_clean.apply(lambda x : 1 if ("cat3" in x["col3"].split(" ")) else(0), axis = 1)
data_clean["cat4"] = data_clean.apply(lambda x : 1 if ("cat4" in x["col3"].split(" ")) else(0), axis = 1)
data_clean["cat5"] = data_clean.apply(lambda x : 1 if ("cat5" in x["col3"].split(" ")) else(0), axis = 1)

def cat6(x):
  if ("CAT6" in x["col3"].split(" ")):
    return True
  elif ("Cat6" in x["col3"].split(" ")):
    return True
  elif ("cat6" in x["col3"].split(" ")):
    return True
  else:
    return False
data_clean["cat6"] = data_clean.apply(lambda x : 1 if (cat6(x)) else(0), axis = 1)

def cat7(x):
  if ("CAT7" in x["col3"].split(" ")):
    return True
  elif ("Cat7" in x["col3"].split(" ")):
    return True 
  elif ("cat7" in x["col3"].split(" ")):
    return True
  elif ("CaT7" in x["col3"].split(" ")):
    return True
  else:
    return False
data_clean["cat7"] = data_clean.apply(lambda x : 1 if (cat7(x))  else(0), axis = 1)


def cat8(x):
  if ("CAT8" in x["col3"].split(" ")) :
    return True
  elif ("Cat8" in x["col3"].split(" ")):
    return True
  elif ("cat8" in x["col3"].split(" ")):
    return True
  else:
    return False
data_clean["cat8"] = data_clean.apply(lambda x : 1 if (cat8(x)) else(0), axis = 1)

data_clean["cat10"] = data_clean.apply(lambda x : 1 if (("cat10" in x["col3"].split(" ")) or ("CAT10" in x["col3"].split(" "))) else(0), axis = 1)
data_clean["cat11"] = data_clean.apply(lambda x : 1 if (("cat11" in x["col3"].split(" ")[1])) else(0), axis = 1)
data_clean["cat12"] = data_clean.apply(lambda x : 1 if (("cat12" in x["col3"].split(" ")[1])) else(0), axis = 1)


data_labeled = copy.deepcopy(data_clean)
data_labeled.drop(columns = ["col3", "col6"], inplace = True)

# Split data
train_index = np.random.choice(data_labeled.index, size = round(len(data_labeled)*0.8)).tolist()
test_index = list(set(data_labeled.index) - set(train_index))

train_data = data_labeled.iloc[train_index, :]
test_data = data_labeled.iloc[test_index, :]

train_data.reset_index(drop=True, inplace = True)
test_data.reset_index(drop = True, inplace = True)

y_col = data_labeled.columns[-11:]
x_col = data_labeled.columns[:-11]

x_train = train_data[x_col]
y_train = train_data[y_col].values

x_test = test_data[x_col]
y_test = test_data[y_col].values

# Label Encoding
cat_cols = ["col7", "col6-2"]

for c in cat_cols:
  le = LabelEncoder()
  le = le.fit(x_train[c])
  x_train[c] = le.transform(x_train[c])
  x_test[c] = le.transform(x_test[c])

''' Model fitting '''
# MLP model training
mlp_multilabel = MLPClassifier(hidden_layer_sizes=(400, 200), max_iter = 1000, random_state = 54)

mlp_multilabel.fit(x_train, y_train)

# Calculate Accuracy
y_pred = mlp_multilabel.predict(x_test)
accuracy = np.sum(np.sum(y_test.astype(int) & y_pred, axis = 1) >0)/y_test.shape[0]