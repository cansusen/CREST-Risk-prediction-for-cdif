import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import random
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV

def location_nonNan(myList):
    i = 0
    n = len(myList)-1
    first_nonNan = -1
    last_nonNan = -1
    while (i<len(myList) and (first_nonNan == -1)):
        if not (np.isnan(myList[i])):
            first_nonNan = i
        i +=1

    while (n>=0 and (last_nonNan == -1)):
        if not (np.isnan(myList[n])):
            last_nonNan = n
        n -=1

    return first_nonNan, last_nonNan


static  = pd.read_csv('Data/staticData.csv', header = 0)
dynamic = pd.read_csv('Data/dynamicData.csv', header = 0)

static  = static.fillna(static.mode().iloc[0])  

label     = static['label']
hadm_id   = static['hadm_id']
columns_to_drop = ['label','hadm_id','discharge_location']
static          = static.drop(columns_to_drop,1)
columns_to_drop = ['day_id']
dynamic         = dynamic.drop(columns_to_drop,1)
dynamic['hadm_id'] = dynamic['hadm_id'].astype(str)

#change categorical variables into binary variables
categorical_features = ["admission_type","admission_location", "insurance","religion","marital_status","ethnicity","gender"]
categorized_data     = pd.get_dummies(static[categorical_features])
static = static.drop(categorical_features,1)
static = pd.concat([static,categorized_data],axis=1,join_axes=[static.index])

    
output_features = []
output_labels   = []
cnt = 0
num_dyn_features = dynamic.shape[1]
p = 5
dynamic_2 = pd.DataFrame()
dynamic_2 = dynamic
dynamic_2 = dynamic_2.fillna(dynamic_2.mode().iloc[0])  

for i in range(0,static.shape[0]):
    patient = hadm_id[i]
    temp_df = dynamic.ix[dynamic.hadm_id == str(patient), :]
    temp_df = temp_df.drop('hadm_id',1)
    temp_df_2 = dynamic_2.ix[dynamic_2.hadm_id == str(patient), :]
    temp_df_2 = temp_df_2.drop('hadm_id',1) 
    n = temp_df.shape[0]
    
    if (n>=p):
        set_of_features = []
        set_of_features.append(static.iloc[i].tolist())
        
        for j in range(0,p):
            set_of_features.append(temp_df_2.iloc[n-j-1].tolist())        
        
        for j in range(0,num_dyn_features-1):
            ts_features = []
            myList = temp_df.iloc[0:n, j].tolist() #Updated so that include all dynamic data when extracting features
            cleanedList = [x for x in myList if str(x) != 'nan']
            #Extract all features on cleanedList
            if cleanedList:
                n_full    = len(myList)
                n_cleaned = len(cleanedList)
                first_nonNan, last_nonNan = location_nonNan(myList)
                np_cleanedTS = np.array(cleanedList)
                np_fullTS = np.array(myList)

                #Features
                nor_lenght      = float(n_cleaned)/n_full
                nor_timeOfFirst = float(first_nonNan+1)/n_full
                nor_timeOfLast  = float(last_nonNan+1)/n_full
                lenght  = n_cleaned
                timeOfFirst = first_nonNan+1
                timeOfLast  = last_nonNan+1
                ts_avg  = np.average(np_cleanedTS)
                ts_std  = np.std(np_cleanedTS)
                ts_max  = np_cleanedTS.max()
                ts_min  = np_cleanedTS.min()
                nor_locMin = (np.nanargmin(np_fullTS)+1)/float(n_full)
                nor_locMax = (np.nanargmax(np_fullTS)+1)/float(n_full)
                locMin = np.nanargmin(np_fullTS)+1
                locMax = np.nanargmax(np_fullTS)+1
                ts_lastVal = myList[last_nonNan]
                ts_lin_avg  = np.average(np_cleanedTS, weights=range(1,len(np_cleanedTS)+1))
                ts_quad_avg = np.average(np_cleanedTS, weights=[x**2 for x in range(1,len(np_cleanedTS)+1)])
                if (n_cleaned>1):
                    ts_absDiff = np.average(np.abs(np.diff(np_cleanedTS)))
                    diffs_01 = [1 if x>0 else -1 for x in np.diff(np_cleanedTS) ]
                    num_increasing = diffs_01.count(1)
                    num_decreasing = diffs_01.count(-1)
                    nor_num_increasing = float(diffs_01.count(1))/len(diffs_01)
                    nor_num_decreasing = float(diffs_01.count(-1))/len(diffs_01)
                    
                    zero_crossings = np.where(np.diff(np.sign(diffs_01)))[0] 
                    flactuation_ratio = float(len(zero_crossings))/len(diffs_01)
                    flactuation_cnt = len(zero_crossings)
                else:
                    ts_absDiff = 0
                    nor_num_increasing = 0
                    nor_num_decreasing = 0
                    num_increasing = 0
                    num_decreasing = 0
                    flactuation_ratio = 0
                    flactuation_cnt = 0
            else:
                nor_lenght      = 0.0
                nor_timeOfFirst = 0.0
                nor_timeOfLast  = 0.0
                lenght      = 0.0
                timeOfFirst = 0.0
                timeOfLast  = 0.0
                ts_avg  = 0.0
                ts_std  = 0.0
                ts_max  = 0.0
                ts_min  = 0.0
                nor_locMin = 0.0
                nor_locMax = 0.0
                locMin = 0.0
                locMax = 0.0
                ts_lastVal = 0.0
                ts_lin_avg  = 0.0
                ts_quad_avg = 0.0
                ts_absDiff  = 0.0
                nor_num_increasing = 0
                nor_num_decreasing = 0
                num_increasing = 0
                num_decreasing = 0
                flactuation_ratio = 0
                flactuation_cnt = 0
          
            ts_features.append(nor_lenght)
            ts_features.append(nor_timeOfFirst)
            ts_features.append(nor_timeOfLast)
            ts_features.append(lenght)
            #ts_features.append(timeOfFirst)
            #ts_features.append(timeOfLast)
            ts_features.append(ts_avg)
            ts_features.append(ts_std)
            ts_features.append(ts_max)
            ts_features.append(ts_min)
            ts_features.append(nor_locMin)
            ts_features.append(nor_locMax)
            #ts_features.append(locMin)
            #ts_features.append(locMax)
            #ts_features.append(ts_lastVal)
            ts_features.append(ts_lin_avg)
            ts_features.append(ts_quad_avg)
            ts_features.append(ts_absDiff)
            ts_features.append(nor_num_increasing)
            ts_features.append(nor_num_decreasing)
            #ts_features.append(num_increasing)
            #ts_features.append(num_decreasing)
            ts_features.append(flactuation_ratio)
            #ts_features.append(flactuation_cnt)
            
            set_of_features.append(ts_features)
            
        features = [item for sublist in set_of_features for item in sublist]
        output_features.append(features)
            
        #Handle labels  
        if (label[i]==0):
            output_labels.append(0)
        else:
            output_labels.append(1)

    

print('# of Observations: ', static.shape[0]-cnt)
print('# of Features:     ', output_features.shape[1])
print(len(output_features))
print(len(output_labels))
print(len(hadm_id))