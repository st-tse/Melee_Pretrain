from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os

from data_module import X_cols, y_cols, split_data

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset', type=str, required = True,
                    help='dataset name')
args = parser.parse_args()

df = pd.read_csv(f"./Data/{args.dataset}.csv", low_memory=False)
df.dropna(axis=0, how='any', inplace=True)

#enc characters
enc_p1 = LabelEncoder()
enc_p2 = LabelEncoder()

enc_p1.fit(df['CHAR_P1'])
enc_p2.fit(df['CHAR_P2'])
df['CHAR_P1'] = enc_p1.transform(df['CHAR_P1'])
df['CHAR_P2'] = enc_p2.transform(df['CHAR_P2'])

df['S_airborne_P1'] = df['S_airborne_P1'].astype(bool).astype(float)
df['S_airborne_P2'] = df['S_airborne_P2'].astype(bool).astype(float)

x_train, y_train, x_test, y_test = split_data(df)

os.chdir('./Datasets/')

pd.DataFrame(x_train).to_csv(f'{args.dataset}_x_train.csv', index=False)
pd.DataFrame(y_train).to_csv(f'{args.dataset}_y_train.csv', index=False)
pd.DataFrame(x_test).to_csv(f'{args.dataset}_x_test.csv', index=False)
pd.DataFrame(y_test).to_csv(f'{args.dataset}_y_test.csv', index=False)
