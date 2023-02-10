import numpy as np
import logging

class Node:
    def __init__(self, attr, attrName, isleaf, label, depth, ig, entr_par_attr, par_attr_val):
        self.attr = attr 
        self.attrName = attrName
        self.children = {}
        self.isleaf = isleaf
        self.label = label
        self.depth = depth
        self.ig = ig 
        self.entr_par_attr = entr_par_attr 
        self.par_attr_val = par_attr_val

    def getAttr(self):
        return self.attr

    def addChild(self, childNode, attrVal):
        self.children[attrVal] = childNode
    
    def predict(self, x):
        if self.isleaf:
            return self.label 
        curVal = x[self.attr]
        if curVal not in self.children.keys():
            return self.label
        return self.children[curVal].predict(x)

    def printNode(self,space=""):
        logging.info(f"{space}Depth: {self.depth}")
        logging.info(f"{space}Feature selected: {self.attrName}")
        logging.info(f"{space}Information Gain for parent feature: {self.ig}")
        logging.info(f"{space}Entropy for parent feature: {self.entr_par_attr}")
        logging.info(f"{space}Parent feature value: {self.par_attr_val}")
        logging.info(f"{space}Label: {self.label}")
        for child in self.children.values():
            child.printNode(space+"\t")


def entropy(counts):
    total = sum(counts) #counts for each class
    H = 0
    for ele in counts:
        p = (ele/total)
        if p!=0:
            H -= p*np.log2(p)
    return H

def infogain(X,Y,attr):
    _, counts = np.unique(Y, return_counts=True)
    entropy_attr = entropy(counts)
    entropy_par = 0
    distinct_attr_values = list(set(X[:,attr]))
    for val in distinct_attr_values:
        indices = np.where(X[:,attr]==val)[0]  
        _, counts = np.unique(Y[indices], return_counts=True)
        entr = entropy(counts)
        entropy_par += (len(indices)/len(Y))*entr
    IG = entropy_attr - entropy_par
    return IG, entropy_attr, entropy_par

class DecisionTree:
    def __init__(self, maxDepth = np.inf):
        self.root = None
        self.depth = 0  
        if maxDepth < 1:
            logging.warning("maxDepth cannot be lower than 1! Setting it to 1.")
            maxDepth = 1
        self.maxDepth = maxDepth
        self.longestPathLen = 0
                
    def _buildtree(self, X, Y, attrNames,attr_list, curDepth, parentInfo={"max_infogain": None, "attr_list[max_attr]": None,"val": None}):
        if curDepth > self.longestPathLen:
            self.longestPathLen = curDepth
        if curDepth >= self.maxDepth or len(attr_list) == 0 or len(np.unique(Y)) == 1:
            vals, counts = np.unique(Y, return_counts=True)
            return Node(None, None, True, vals[np.argmax(counts)],curDepth, parentInfo["max_infogain"], parentInfo["attr_list[max_attr]"], parentInfo["val"])

        max_infogain = -1  
        max_attr = None  
        i = 0 
        for attr in attr_list:
            ig,entropy_attr,entropy_par= infogain(X,Y,attr)
            if ig > max_infogain:
                max_infogain = ig
                max_attr = i
                ent = entropy_par
            i += 1
        
        
        vals, counts = np.unique(Y, return_counts=True)
        root = Node(attr_list[max_attr],attrNames[attr_list[max_attr]],False,vals[np.argmax(counts)],curDepth, parentInfo["max_infogain"], parentInfo["attr_list[max_attr]"], parentInfo["val"])
        
        attrVals = np.unique(X[:, attr_list[max_attr]])
        new_attr_list = np.delete(attr_list,max_attr)
        for val in attrVals:
            inds = np.where(X[:, attr_list[max_attr]] == val)[0]
            if len(inds)==0:
                root.addChild(Node(None, None, True, vals[np.argmax(counts)], curDepth+1, max_infogain, attr_list[max_attr], val),curDepth)
            else:
                parentInfo = {
                    "max_infogain": max_infogain,
                    "attr_list[max_attr]": ent,
                    "val": val
                }
                root.addChild(self._buildtree(X[inds],Y[inds], attrNames, new_attr_list, curDepth+1, parentInfo), val) #S_v (X[inds],Y[inds])
        return root

    def buildtree(self, X, Y, attrNames, attr_list=[]):
        if len(attr_list) == 0:
            attr_list = np.arange(X.shape[1])
        self.root = self._buildtree(X, Y, attrNames, attr_list, 0)


    def predict(self, X):
        preds = []
        for x in X:
            preds.append(self.root.predict(x))
        return preds

    def getLongestPathLen(self):
        return self.longestPathLen

    def getRootFeature(self):
        if self.root:
            return self.root.getAttr()
        return None

    def printTree(self):
        self.root.printNode("")
        