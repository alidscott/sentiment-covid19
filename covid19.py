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

# df[6] = tweets
# df[7] = labels
df = pd.read_csv("output.csv", delimiter=',', header=None)

# Load the pretrain model
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

##Want BERT instead of distilBERT? Uncomment the following line:
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer,
# 'bert-base-uncased')

#pr Load pre-trained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


# TOKENIZING
tokenized = df[6].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


# Padding the tokenizing list so they're all the same size
# so it can be processed by BERT all at once
max_len =0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
np.array(padded).shape


# MASKING to ignore the padding we added when processing the input
attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

# Run the input through the model
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

# slice the first token of the output which is added by BERT. It is a classification
# token that is added at the begining of every sentence

features = last_hidden_states[0][:,0,:].numpy()
labels = df.iloc[:,7] #indicates positive or negative


# MODEL 2 :  TRAIN TEST and SPLIT
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

# Logistic regression
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

# Evaluate the model
print("The accuracy of our model: ",  lr_clf.score(test_features, test_labels))
