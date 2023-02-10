import csv
import argparse
import numpy as np
import logging
from classes import DecisionTree, infogain  

#logging.basicConfig(filename='1_2_Full_tree_implementation.txt',filemode="w",level=logging.DEBUG)
#logging.basicConfig(filename='3_Limiting_Depth_k_fold_cross_validation_log.txt',filemode="w",level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()

parser.add_argument(
    "-train",
    help="Path to train file",
    default="./data/q1_train.csv"
)

parser.add_argument(
    "-test",
    help="Path to test file",
    default="./data/q1_test.csv"
)

parser.add_argument(
    "-valid",
    help="Path to directory containing validation sets",
    default="./data/CVfolds/"
)

parser.add_argument(
    "-k",
    "--numFolds",
    type=int,
    help="No. of folds for k-fold cross validation",
    default = 5,
)

parser.add_argument(
    "-kfcv",
    "--kFoldCrossValidation",
    action="store_true",
    help="Boolean flag to enable k fold cross validation"
)


args = parser.parse_args()

train = args.train
test = args.test
valid = args.valid
numFolds = args.numFolds
kfcv = args.kFoldCrossValidation

with open(test, "r") as f:
    test = list(csv.reader(f, delimiter=","))
test = np.array(test[1:])

x_test = test[:,:-1]
y_test = test[:, -1]

if not kfcv:
    logging.info("\n################### # 1 & 2. Baseline and Implementation: Full trees ###################################\n")
    with open(train, "r") as f:
        train = list(csv.reader(f, delimiter=","))
    headers = train[0]
    train = np.array(train[1:])

    x_train = train[:,:-1]
    y_train = train[:, -1]
    vals, counts = np.unique(y_train, return_counts=True)
    logging.info(f"")
    logging.info(f" Most frequent label in train dataset: {vals[np.argmax(counts)]}")
    logging.info(f" The training Accuracy for classifier which always predicts maxLabel: {np.round((np.max(counts)/len(y_train))*100, 2)}%")
    
    vals1, counts1 = np.unique(y_test, return_counts=True)
    max_label = vals[np.argmax(counts)] 
    ind1 = np.where(vals1 == max_label)
    cnts1 = counts1[ind1]
    logging.info(f" The test Accuracy for classifier which always predicts maxLabel: {np.round((cnts1/len(y_test))*100, 2)}%")

    t = DecisionTree()
    t.buildtree(x_train, y_train, headers)
    t.printTree()
    preds = t.predict(x_train)
    acc = np.sum(y_train == preds)/len(preds)
    logging.info(f"\t Train set accuracy: {np.round(acc*100, 2)}%")
    preds = t.predict(x_test)
    acc = np.sum(y_test == preds)/len(preds)
    logging.info(f"\t Test set accuracy: {np.round(acc*100, 4)}%")
    logging.info(f"\t The root feature that is selected by algorithm : {headers[t.getRootFeature()]}")
    logging.info(f"\t Information gain for the root feature: {np.round(infogain(x_train, y_train, t.getRootFeature())[0],5)}")
    logging.info(f"\t The max-depth of the tree is : {t.getLongestPathLen()}")

else:
    logging.info("\n###################### 3. Limiting Depth - (For 5-fold cross-validation) ###################################")
    bestAcc = 0   
    bestDepth = None
    for depth in range(1,6):
        logging.info(f"")
        logging.info(f" ************** Depth: {depth} ************* ")
        avg_acc = 0
        all_acc =[]
        for i in range(1,numFolds+1):
            logging.info(f"\tChoosing fold {i} as validation set")
            with open(valid+"fold"+str(i)+".csv", "r") as f:
                validation = list(csv.reader(f, delimiter=",")) 
            headers = validation[0]
            validation = np.array(validation[1:])
            x_valid = validation[:,:-1]
            y_valid = validation[:, -1]

            tr = []
            for j in range(1, numFolds+1):
                if i!=j:
                    with open(valid+"fold"+str(i)+".csv", "r") as f:
                        data = list(csv.reader(f, delimiter=","))
                    tr.extend(data[1:])
            tr = np.array(tr)

            x_train = tr[:,:-1]
            y_train = tr[:, -1]
            
            t = DecisionTree(depth)
            t.buildtree(x_train, y_train, headers)
            preds = t.predict(x_train)
            acc = np.sum(y_train == preds)/len(preds)
            all_acc.append(acc)

            avg_acc += acc
            logging.info(f"\t\tTrain set accuracy: {np.round(acc*100, 2)}%")
                
            preds = t.predict(x_valid)
            acc = np.sum(y_valid == preds)/len(preds)
            logging.info(f"\t\tValid set accuracy: {np.round(acc*100, 2)}%")

        avg_acc /= numFolds
        if avg_acc > bestAcc:
            bestAcc = avg_acc
            bestDepth = depth

        std = 0

        for i in range(len(all_acc)):
            std += (avg_acc-all_acc[i])**2
        
        std /= numFolds
        std = np.sqrt(std)
        
        logging.info(f"")
        logging.info(f"\t Average cross validation accuracy: {np.round(avg_acc*100, 2)}%")
        logging.info(f"\t Standard deviation for depth {depth} : {np.round(std,5)}")


    logging.info(f"")
    logging.info(f"\t Depth {bestDepth} has the best accuracy of {np.round(bestAcc*100,2)}% and is thus the best choice of depth")


    with open(train, "r") as f:
        train = list(csv.reader(f, delimiter=","))
    headers = train[0]
    train = np.array(train[1:])

    x_train = train[:,:-1]
    y_train = train[:, -1]

    t = DecisionTree(bestDepth)
    t.buildtree(x_train, y_train, headers)
    preds = t.predict(x_test)
    acc = np.sum(y_test == preds)/len(preds)
    logging.info(f"\t Test set accuracy using the best depth : {np.round(acc*100, 4)}%")