import pandas as pd
import numpy as np
import logging
import csv 
from classes import DecisionTree, infogain

#logging.basicConfig(filename='4_imputations_log.txt',filemode="w",level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

logging.info(" ######################################### Imputations ##################################################")
logging.info(" ################### Imputed the missing feature values, and then trained the full decision tree on that data. ################")


df_train = pd.read_csv("./data_missing/train.csv")
df_test = pd.read_csv("./data_missing/test.csv")

column_headers = list(df_train.columns.values)

#print(df_test.head())
#print("The Column Header :", column_headers)

#print(f"No of missing values before performing any imputation for trainset :\n{df_train.eq('?').sum()}")
#print(f"No of missing values before performing any imputation for testset :\n{df_test.eq('?').sum()}")
    
'''
print(df_train.info)
inds = df_train.isin(["?"]).any(1)
df_train = df_train[~inds]
'''
#print(df_train.eq("?").any().any())

for col in column_headers:
    if df_train[col].eq('?').any():
        col_mode = df_train[col].mode()[0]
        df_train[col] = df_train[col].replace('?', col_mode)



#print(f"No of missing values after performing any imputation for trainset :\n{df_train.eq('?').sum()}")
#print(f"No of missing values after performing any imputation for testset :\n{df_test.eq('?').sum()}")


df2_train = pd.read_csv("./data_missing/train.csv")
df2_test = pd.read_csv("./data_missing/test.csv")

#print(f"No of missing values before performing any imputation for trainset df2_train :\n{df2_train.eq('?').sum()}")
#print(f"No of missing values before performing any imputation for testset df2_train :\n{df2_test.eq('?').sum()}")

for col in df2_train.columns:
    for label in df2_train[df2_train.columns[-1]].unique():
        label_rows = df2_train[df2_train.columns[-1]] == label
        mode = df2_train.loc[label_rows, col].mode()[0]
        missing_values = df2_train[col] == "?"
        df2_train.loc[missing_values & label_rows, col] = mode

#print(f"No of missing values after performing any imputation for trainset df2_train :\n{df2_train.eq('?').sum()}")
#print(f"No of missing values after performing any imputation for testset df2_train :\n{df2_test.eq('?').sum()}")

def train_on_full_decision_tree(train,test):
    headers = column_headers

    train = np.array(train[1:])
    test  = np.array(test[1:])

    x_train = train[:,:-1]
    y_train = train[:, -1]

    x_test = test[:,:-1]
    y_test = test[:, -1]

    t = DecisionTree()
    t.buildtree(x_train, y_train, headers)
    t.printTree()
    preds = t.predict(x_train)
    acc = np.sum(y_train == preds)/len(preds)
    logging.info(f"\t Train set accuracy: {np.round(acc*100, 2)}%")
        
    preds = t.predict(x_test)
    acc = np.sum(y_test == preds)/len(preds)
    logging.info(f"\t Test set accuracy: {np.round(acc*100, 10)}%")


    logging.info(f"\t The root feature that is selected by algorithm : {headers[t.getRootFeature()]}")
    logging.info(f"\t Information gain for the root feature: {infogain(x_train, y_train, t.getRootFeature())[0]}")
    logging.info(f"\t The max-depth of the tree is : {t.getLongestPathLen()}")



def most_common_fv_whole_dataset():   
    train1 = np.array(df_train[1:])
    test1  = np.array(df_test[1:])
    train_on_full_decision_tree(train1,test1)


def most_common_fv_for_same_label():   
    train = np.array(df2_train[1:])
    test  = np.array(df2_test[1:])
    train_on_full_decision_tree(train,test)

logging.info("")
logging.info(" ######################### Replacing it with the most common feature value in the whole dataset ###################################")
logging.info("")

most_common_fv_whole_dataset()

logging.info("")
logging.info(" ##################### Replacing it with the most common feature value for the same label ###################################")
logging.info("")

most_common_fv_for_same_label()

