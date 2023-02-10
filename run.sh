#Experiments
# 1 & 2. Baseline and Implementation: Full trees
python3 ./main.py

# 3. Limiting Depth - (For 5-fold cross-validation)
python3 ./main.py -kfcv

# 4. Imputations - (In order to Impute the missing feature values, and then train the full decision tree on that data)
python3 ./imputations.py

#included all the 3 outputs in 3 different log files. - 1_2_Full_tree_implementation.txt ,3_Limiting_Depth_k_fold_cross_validation_log.txt
# and 4_imputations_log files.
