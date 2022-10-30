import os
import cv2
import random
import cv2
import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def writeFeatures(features, flag, folder, name):
  if(flag):
    features.to_csv(os.path.join(folder,
                                 name),
                    mode='a',
                    header=True,
                    index=False)
    flag = False
  else:
    features.to_csv(os.path.join(folder,
                                 name),
                  mode='a',
                  header=False,
                  index=False)
  return flag

# get a random sample of images from the train folder of challenge 1
# input:  path of the folder with train and test subfolders
#         amount of elements you want to sample, preferable even
#         percentage of train from the total amount
#         percentage of test from the total amount
# output: list of shuffle samples from training and samples from test
def get_sample_ch2(path = "/home/emily/Desktop/CAD/challenge2/train", output="", flag=True, cbcc=1993, cmel=2713, cscc=376):
    
    dictF = {}
    features = pd.DataFrame(dtype=np.float64)
    
    _, _, bcc = next(os.walk(os.path.join(path, "bcc")))
    _, _, mel = next(os.walk(os.path.join(path, "mel")))
    _, _, scc = next(os.walk(os.path.join(path, "scc")))
    
    bcc = [ os.path.join(path,'bcc', item) for item in bcc ]
    mel = [ os.path.join(path,'mel', item) for item in mel ]
    scc = [ os.path.join(path,'scc', item) for item in scc ]
    
    # if output file exists
    if os.path.exists(output):
        # no header when writing after
        flag = False
        # read file
        outputfile = pd.read_csv(output)    
        # get names already run
        names = outputfile["name"]
        # divide the names in the two categories
        
        doneBcc = []
        doneMel = []
        doneScc = []
        all_names = names.values.tolist()
        
        for name in names.values.tolist():
            # element  not nan
            if isinstance(name, str):
                if "bcc" in name:
                    doneBcc.append(name)
                if "mel" in name:
                    doneMel.append(name)
                if "scc" in name:
                    doneScc.append(name)
                    
        # getting the pending nevus and others files names
        pendingBcc = list(set(bcc) - set(doneBcc))
        pendingMel = list(set(mel) - set(doneMel))
        pendingScc = list(set(scc) - set(doneScc))
        
        randBcc = random.sample(pendingBcc, k=(cbcc-len(pendingBcc)))
        randMel = random.sample(pendingMel, k=(cmel-len(pendingMel)))
        randScc = random.sample(pendingScc, k=(cscc-len(pendingScc)))

    else:
        randBcc = random.sample(bcc, k=int(cbcc))
        randMel = random.sample(mel, k=int(cmel))
        randScc = random.sample(scc, k=int(cscc))
              
    
    subbcc = [ os.path.join(path,'bcc', item) for item in randBcc ]
    submel = [ os.path.join(path,'mel', item) for item in randMel ]
    subscc = [ os.path.join(path,'scc', item) for item in randScc ]
    
    samples = [*subbcc, *submel, *subscc]
    
    np.random.shuffle(samples)
    print(len(samples))
    return samples, flag

# get a random sample of images from the train folder of challenge 1
# input:  path of the folder with train and test subfolders
#         amount of elements you want to sample, preferable even
#         percentage of train from the total amount
#         percentage of test from the total amount
# output: list of shuffle samples from training and samples from test
def get_sample(path = "/home/emily/Desktop/CAD/challenge1/train", output="", flag=True, amount=1000):
    
    # initial variables
    dictF = {}
    features = pd.DataFrame(dtype=np.float64)
    
    # get all names
    _, _, nevus = next(os.walk(os.path.join(path, "nevus")))
    _, _, others = next(os.walk(os.path.join(path, "others")))
    
    nevus = [ os.path.join(path,'nevus', item) for item in nevus ]
    others = [ os.path.join(path,'others', item) for item in others ]
    
    # if output file exists
    if os.path.exists(output):
        # no header when writing after
        flag = False
        # read file
        outputfile = pd.read_csv(output)    
        # get names already run
        names = outputfile["name"]
        # divide the names in the two categories
        
        doneNevus = []
        doneOthers = []
        all_names = names.values.tolist()
        
        for name in names.values.tolist():
            # element  not nan
            if isinstance(name, str):
                if "nevus" in name:
                    doneNevus.append(name)
                if "others" in name:
                    doneOthers.append(name)
                    
        # getting the pending nevus and others files names
        pendingNevus = list(set(nevus) - set(doneNevus))
        pendingOthers = list(set(others) - set(doneOthers))
        # 
        print(abs(int(amount/2) - len(doneNevus)))
        print(abs(int(amount/2) - len(doneOthers)))
        randnevus = random.sample(pendingNevus, k=abs(int(amount/2) - len(doneNevus)))
        randothers = random.sample(pendingOthers, k=abs(int(amount/2) - len(doneOthers)))

    else:
        randnevus = random.sample(nevus, k=int(amount/2))
        randothers = random.sample(others, k=int(amount/2))
              
    subnevus = [ os.path.join(path,'nevus', item) for item in randnevus ]
    subothers = [ os.path.join(path,'others', item) for item in randothers ]

    samples = [*subnevus, *subothers]
    
    np.random.shuffle(samples)
    print("samples ", len(samples))
    return samples, flag



# hair removal method base on bottom hat 
#https://github.com/MujtabaAhmad0928/SCRS/blob/fe2f6a7ed9fd1897f8c47da509b2c8a399ea4910/preprocessing.py
def hair_removal_BH(matrix, kernel_size = 17):
    # grayscale
    gray = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY )
    # blackhat 
    kernel = cv2.getStructuringElement(1,(kernel_size,kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # threshold
    ret,th = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # impainting
    output = cv2.inpaint(matrix, th, 1, cv2.INPAINT_TELEA)
    return output

# RELEVANT
# hairs near the edge of the image are not fully removed



#https://medium.com/@er_95882/colour-vision-lands-experiments-with-colour-constancy-white-balance-and-examples-in-python-93a71d0c4cbe
def grey_world(image):
    image = image / 255.

    pWhite = 0.05
    
    # In OpenCV the channels order is Blue-Green-Red
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]

    red = red / np.mean(red)
    green = green / np.mean(green)
    blue = blue / np.mean(blue)

    red_sorted = sorted(red.ravel())
    green_sorted = sorted(green.ravel())
    blue_sorted = sorted(blue.ravel())

    total = len(red_sorted)

    max_index = int(total * (1. - pWhite))
    image[:, :, 2] = red / red_sorted[max_index]
    image[:, :, 1] = green / green_sorted[max_index]
    image[:, :, 0] = blue / blue_sorted[max_index]
    
    return image

#https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images
def clahe_rgb(img, gridsize=100):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #lab_planes = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))

    #lab_planes = np.array([map(list, lab_planes)])
    
    lab[0] = clahe.apply(lab[0])

    #lab = cv2.merge(lab_planes)

    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return result

# segmentation
def segmentation_kmeans(hairless):
    # color space change
    hairless = cv2.cvtColor(output_bh, cv2.COLOR_BGR2RGB)
    # reshape the image to be a list of pixels
    data = hairless.reshape((hairless.shape[0] * hairless.shape[1], 3))
    data = np.float32(data)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness,labels,centers = cv2.kmeans(data,2,None,criteria,10,flags)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res2 = res.reshape((hairless.shape))

    gray = cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY)

    mask = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return mask

#################### MACHINE LEARNING ####################

# random forest

from sklearn.ensemble import RandomForestClassifier

def RandomForest(X_train, y_train, cv=5, best_params = dict()):
  if len(best_params) == 0:
    print("Searching for best hyperparameters")
    params = {'criterion': ['gini'],
              'n_estimators': [100, 500, 900], # , 500, 900
    #          'max_features': ['sqrt'],#, 'sqrt', 'log2'
              'max_depth' : [10, 12]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs = -1), param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    print("The best parameters for rf are %s with an accuracy of %0.4f"%(best_params, grid.best_score_))
  
  classifier = RandomForestClassifier(random_state=42, n_jobs = -1)
  
  return classifier, best_params


from sklearn.svm import SVC
def SVC_linear(X_train, y_train, cv=5, best_params = dict()):

  if len(best_params) == 0:
    lower_value_C = 1
    higher_value_C = 10
    n_values = 10
    base = 10
    params = {'C': [1, 3, 5,9.11], 
              'kernel' : ['rbf'],
              'gamma': [2.5, 5, 10]}

    grid = GridSearchCV(SVC(random_state = 42, probability=True), param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    print("The best parameters for svm are %s with an accuracy of %0.4f"%(best_params, grid.best_score_))
  
  classifier = SVC(random_state = 42, probability=True)

  return classifier, best_params 

from sklearn.ensemble import AdaBoostClassifier 

def AdaBoost(X_train, y_train, cv=5, best_params = dict()):
  if len(best_params) == 0:
    params = {'n_estimators': [300, 600 , 900],
              'learning_rate': [0.1, 0.5, 1]}
    grid = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    print("The best parameters for ab are %s with an accuracy of %0.4f"%(best_params, grid.best_score_))
  
  classifier = AdaBoostClassifier(random_state=42)

  return classifier, best_params


from sklearn.ensemble import GradientBoostingClassifier # gradient boosting regressor
# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
def GradientBoosting(X_train, y_train, cv=5, best_params = dict()):
  if len(best_params) == 0:
    params = {'learning_rate': [0.05, 0.1, 0.2],#0.05, 0.2
              #'min_samples_split': [0.5, 0.8],
              #'min_samples_leaf': [0.1, 0.2, 0.5],
              'max_depth':[8],
              #'max_features':['sqrt'],#'log2'
              #'criterion': ['friedman_mse',  'mae'],
              #'subsample':[0.5, 1.0],
              'n_estimators':[300, 600]}
    grid = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    print("The best parameters for gb are %s with an accuracy of %0.4f"%(best_params, grid.best_score_))

  classifier = GradientBoostingClassifier(random_state=42)
  
  return classifier, best_params


# k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

def knn(X_train, y_train, cv=5, best_params = dict()):
  if len(best_params) == 0:
    params = {'n_neighbors':[3,5]}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

  classifier = KNeighborsClassifier()

  return classifier, best_params



# fit report
def fit_report(pipe, X_train, y_train, X_test, y_test, pipelineName='', classifierName='', balancingType = ''):
    # To run this without problems, you need to define RESULTS_DIR, which is the path where the models and the confusion matrix will be
    # kept.
    
    
    print("*****************************************************")
    
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test) 
    
    print(" ### Report ###")
    print(classification_report(y_test, pred))
    
    print(" ### score ###")
    print(pipe.score(X_test, y_test))

    print(" ### accuracy ###")
    acc = accuracy_score(y_test, pred)
    print(acc)

    print("### f1_score ###")
    f1 = f1_score(y_test, pred, average='weighted')
    print(f1) # 1 is best
    
    print("### confusion matrix ###")
    print(confusion_matrix(y_test, pred)) # diagonal stronger
    
    df_cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(df_cm, annot=True, fmt='d') # font size
    plt.savefig(os.path.join(RESULTS_DIR, f'confmat_{pipelineName}_{classifierName}_{balancingType}.png'), format='png')

    # Save the final model with its respective fold
    filename = f'model_{classifierName}_{pipelineName}_{balancingType}.pkl'
    pickle.dump(pipe, open(os.path.join(RESULTS_DIR, filename), 'wb'))
    print(f"Model {pipelineName} of {classifierName} was saved!")  
    
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    return acc, f1
