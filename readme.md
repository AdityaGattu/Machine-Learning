
#Experiments
# To run shell script - sh run.sh

# python3 ./main.py  
-> You can execute this line alone to see the o/p of 1 & 2. Baseline and Implementation: Full trees 
 uncommenting the other lines in run.sh file

# python3 ./main.py -kfcv  
-> For # 3. Limiting Depth - (For 5-fold cross-validation)

4.Imputations - (In order to Impute the missing feature values, and then train the full decision tree on that data)
# python3 ./imputations.py

# ALL the  outputs for each case are included in 3 different log files. 
# 1_2_Full_tree_implementation.txt , 3_Limiting_Depth_k_fold_cross_validation_log.txt
# and 4_imputations_log files.

Also, Can uncomment the lines -  logging.basicConfig lines in main.py,impuations.py and run the shell script with necessary commands to log the output into these into log files.