#create file
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#GetOldTweets3 --querysearch "corona virus" --lang en --maxtweets

df = pd.read_csv("output_got.csv", delimiter=',', header=None)
tweets = df[6]
print(tweets)
