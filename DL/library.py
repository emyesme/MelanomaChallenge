import os
import cv2
import random
import numpy as np 
import pandas as pd

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


def get_column_names(dataframe):
    #returns the column name as values
    
    return dataframe['name'].values.tolist()
    

def uniques_names(entire_list, subset):
    # Given a list of names take the ones that are not part of that list
    not_used = [i for i in entire_list if i not in subset]
    
    return not_used


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
