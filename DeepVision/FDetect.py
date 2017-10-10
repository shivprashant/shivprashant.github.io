
# coding: utf-8

# In[1]:

#Prepare the data
#Convert images to data frame and add labels

import os as os
import pandas as pd
import numpy as np
root_image="C:/Users/shivsood/OneDrive/DeepVision/Images"
training_images=""
training_images= root_image + "/TrainingImages"
training_images_path= root_image + "/TrainingImages"


from MyLib import working_image_size
from MyLib import loadImageAndLabel, imgFromArray


import matplotlib
import matplotlib.pyplot as plt
import math
    
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="upper left")

def trainRF(x_train,y_train,x_test,y_test):
    # Train
    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(x_train,y_train)    
       
    y_pred = rf_clf.predict(x_test)    

    #from sklearn.metrics import precision_recall_curve

    #precisions, recalls, thresholds = precision_recall_curve(y_test,y_scores)
        

    #plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    #plt.show()

    
    #from sklearn.metrics import roc_curve, roc_auc_score

    #fpr, tpr, thresholds = roc_curve(y_test,y_scores)
    #plot_roc_curve(fpr, tpr, thresholds)
    #plt.show()

    
    #y_pred_by_s = y_scores > -5.3e+07
    from MyLib import classScoreMetric,cfMatrix
    print(cfMatrix(y_pred,y_test))
    accuracy, pscore, rscore, f1score = classScoreMetric(y_pred,y_test)
    #print(accuracy, pscore, rscore, f1score)
    resultRec = {}
    resultRec['Algo'] = 'Standard RF'
    resultRec['Accuracy'] = accuracy
    resultRec['PrecisionScore'], resultRec['RecallScore'],resultRec['f1score']=pscore, rscore, f1score
    #resultRec['roc_auc_score'] = roc_auc_score(y_test,y_scores)

    return resultRec

def trainSDG(x_train,y_train,x_test,y_test,decisionscore=0):
    # Train
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(x_train,y_train)    
    
    
    y_scores = sgd_clf.decision_function(x_test)
    #y_pred_by_s = y_scores > -5.3e+07
    #y_pred_by_s = y_scores >  1.04e+09


    #y_pred = y_pred_by_s
    y_pred = sgd_clf.predict(x_test)
    #imgFromArray(x_train,y_train)
    #imgFromArray(x_test,y_test)
    
    
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_test,y_scores)
    #print(precisions, recalls, threshold)

      

    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.show()

    
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, thresholds = roc_curve(y_test,y_scores)
    plot_roc_curve(fpr, tpr, thresholds)
    plt.show()

    
    #y_pred_by_s = y_scores > -5.3e+07
    from MyLib import classScoreMetric,cfMatrix
    print(cfMatrix(y_pred,y_test))
    accuracy, pscore, rscore, f1score = classScoreMetric(y_pred,y_test)
    #print(accuracy, pscore, rscore, f1score)
    resultRec = {}
    resultRec['Algo'] = 'SGD Classifier'
    resultRec['Accuracy'] = accuracy
    resultRec['PrecisionScore'], resultRec['RecallScore'],resultRec['f1score']=pscore, rscore, f1score
    resultRec['roc_auc_score'] = roc_auc_score(y_test,y_scores)

    return resultRec

    
    

print("Reading Training data...")
df_data, df_label,count_of_training_images = loadImageAndLabel(training_images_path)
print("Read ", count_of_training_images, "Training files")

cv_TotalFolds = 2
cv_FoldNumber = 0
skFolds = StratifiedKFold(n_splits=cv_TotalFolds,random_state=42)

resultsdf = pd.DataFrame()

allLabels = set(df_label)

for train_index, test_index in skFolds.split(df_data[0:count_of_training_images],df_label[0:count_of_training_images]):
    
    x_train, y_train = df_data[train_index],df_label[train_index]   
    x_test, y_test  =  df_data[test_index],df_label[test_index]
    
    cv_FoldNumber = cv_FoldNumber + 1

    for label in allLabels:
        resultRec = trainSDG(x_train,y_train==label,x_test,y_test==label)
        resultRec['CurrentLabel'] = label
        resultRec['CV_TotalFolds'], resultRec['CV_FoldNumber']  = cv_TotalFolds, cv_FoldNumber
        resultsdf = resultsdf.append(resultRec,ignore_index=True)

        resultRec = trainRF(x_train,y_train==label,x_test,y_test==label)
        resultRec['CurrentLabel'] = label
        resultRec['CV_TotalFolds'], resultRec['CV_FoldNumber']  = cv_TotalFolds, cv_FoldNumber
        resultsdf = resultsdf.append(resultRec,ignore_index=True)


#Create a CSV file

resultsdf.to_csv("data/DeepVisionResults.csv",sep=",")

#print(resultsdf)




    

    
    
      
 
            




