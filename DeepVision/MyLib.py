import numpy as np
import os as os

working_image_size = (100,100)

def loadImages(dir_path):
    import cv2
    #Reads a directory of images and returns set of images stored as a data frame
    #returns a 2D numpy array of images. format - File_Name, <One Column Per Pixel>
    
    da = np.empty([100,working_image_size[0]*working_image_size[1]])
    counter = 0
    
    for f in os.listdir(dir_path):
        print(f)
        if(f.endswith(".JPG")):
            filePath= dir_path +"/" + f
            #print("Found JPG file ",filePath)
            img = cv2.resize(cv2.imread(filePath,cv2.IMREAD_GRAYSCALE),working_image_size)
            #print("Shape before reshape",img.shape)
            img = img.reshape(-1,working_image_size[0]*working_image_size[1])
            da[counter] = img[0]
            counter = counter + 1
            #print("Shape after reshape",img.shape)
            #print("Shape after reshape",img.head(1))  
    
    #print("Shape after reshape",da.shape)
    #print("Shape after reshape",da)
    return da,counter

def createLabel(label,nr):
    print(label)
    
    labels = [label for i in range(0,nr)]
    df_label = np.array(labels)
    #df_label = df_label.reshape(nr,-1)
    
      
    return df_label

#Load images and create labels from Image titles
def getLabelFromFileName(filename):
    l= filename.split('__')
    
    if(len(l) == 0):
        print("error")
        label = "Unknow"
    else:
        label = l[0]
    
    #print("file ", filename, "label ", label)
    return label
    

def loadImageAndLabel(dir_path):
    import cv2
    labelList = []
    counter = 0
    da = np.empty([100,working_image_size[0]*working_image_size[1]])
    
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path,f))]
    
    for f in files:
        #print("processing file ",f )
        fp = dir_path + "/" + f
        
        #Append label
        labelList.append(getLabelFromFileName(f))
        
        #Load file and store as single rom in the data array
        img = cv2.resize(cv2.imread(fp,cv2.IMREAD_GRAYSCALE),working_image_size)
        img = img.reshape(-1,working_image_size[0]*working_image_size[1])
        da[counter] = img[0]
        counter = counter + 1
    
    #Basic sanity check
    if(len(labelList)!=counter):
        print("Data reading problem, please check")

    dl = np.array(labelList)
    
    return da,dl,counter 

#Create statified folds on data, train a model on each of the cross folds and Calculate Performance metric

def cfMatrix(pred_label,actual_label):
    import pandas as pd
    y_actu = pd.Series(actual_label, name='Actual')
    y_pred = pd.Series(pred_label, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    return df_confusion

def classScoreMetric(y_pred,y_test):
   
    
    accuracy =round(100*sum(y_pred[y_test])/len(y_test),2) 
    
    from sklearn.metrics import precision_score
    pscore =  round(precision_score(y_test,y_pred),2)      
    
    from sklearn.metrics import recall_score
    rscore = round(recall_score(y_test,y_pred),2)
    
    from sklearn.metrics import f1_score
    f1score = round(f1_score(y_test,y_pred),2)
    
    return accuracy, pscore, rscore, f1score

        
def predictionAccuracy(y_pred,y_test):
    for label in set(y_test):
        print("Prediction accuracy for label " , label, "is : ", sum(y_pred[y_test==label]==label)/len(y_pred[y_test==label])*100)
  

def plot_roc_curve(fpr, tpr, label=None):
            import matplotlib.pyplot as plt
            plt.plot(fpr, tpr, linewidth=2, label=label)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.axis([0, 1, 0, 1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            
def cfMatrixPerLabel(y_pred,y_test):
    import pandas as pd
    print("==== Confusion Matrix ==== ")
    
    labels = set(y_test)
    print(labels)
    for label in labels:
        print("** For Label ", label, " **")
        y_as = pd.Series(y_test==label, name='Actual')
        y_ps = pd.Series(y_pred==label, name='Predicted')
        df_confusion = pd.crosstab(y_as, y_ps)
        print(df_confusion)
        print("")
        

    
    

    
  
       
   
def imgFromArray(arrOfImages,labels):
   
    import matplotlib
    import matplotlib.pyplot as plt
    import math
    
    x = math.ceil(math.sqrt(len(arrOfImages)))
    fig, axes = plt.subplots(nrows=x, ncols=x)

    for index in range(0,len(arrOfImages)):
        ax = plt.subplot(x, x, index+1)
        ar = arrOfImages[index]
        ar_s = ar.reshape(working_image_size)
        plt.imshow(ar_s, cmap = matplotlib.cm.binary,
               interpolation="nearest")
        ax.annotate(labels[index], fontsize=20,color='red' ,xy=(50, 50))
        plt.axis("off")
    
    plt.show()