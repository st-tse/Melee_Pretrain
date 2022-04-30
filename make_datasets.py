from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os

from data_module import split_data

import argparse

pd.options.mode.chained_assignment = None  # default='warn'

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset', type=str, required = True,
                    help='dataset name')
parser.add_argument('-t','--type', choices = ['b','s'], required = True,
                    help='type of dataset')
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

try:
    df['S_airborne_P1'] = df['S_airborne_P1'].astype(bool).astype(float)
    df['S_airborne_P2'] = df['S_airborne_P2'].astype(bool).astype(float)
except:
    df['S_airborne_P1_x'] = df['S_airborne_P1_x'].astype(bool).astype(float)
    df['S_airborne_P2_x'] = df['S_airborne_P2_x'].astype(bool).astype(float)
    df['S_airborne_P1_y'] = df['S_airborne_P1_y'].astype(bool).astype(float)
    df['S_airborne_P2_y'] = df['S_airborne_P2_y'].astype(bool).astype(float)


x_train, y_train, x_test, y_test = split_data(df, dataset_type=args.type)

os.chdir('./Datasets/')

pd.DataFrame(x_train).to_csv(f'{args.dataset}_x_train.csv', index=False)
pd.DataFrame(y_train).to_csv(f'{args.dataset}_y_train.csv', index=False)
pd.DataFrame(x_test).to_csv(f'{args.dataset}_x_test.csv', index=False)
pd.DataFrame(y_test).to_csv(f'{args.dataset}_y_test.csv', index=False)
