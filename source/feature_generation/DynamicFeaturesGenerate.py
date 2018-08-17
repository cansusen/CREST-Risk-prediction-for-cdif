import csv
import numpy as np
import pandas as pd
from dateutil import parser
from datetime import timedelta
from datetime import datetime

##############################################################################
### Cansu Sen
### Date: 26-July-2016
### Generate dynamic features
### Input: CSV files generated through MIMIC III database from the following tables:
###        Chartevents, Labevents, Microbiologyevents, Services, Medications
### Output: Dynamic feature matrix
### MIMIC III is a publicly available critical care dataset. We are not at the
### liberty of sharing the exact data files we used, however, one can obtain 
### access to MIMIC dataset and create these files. 
### MIMIC III - https://mimic.physionet.org
##############################################################################

print ('Start time: ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

p_static_features    = pd.read_csv('Data/info_positive.csv',sep=',')
p_dynamic_features   = pd.read_csv('Data/dyn_lab_longer_cdiff.csv',sep=',')
p_dynamic_features_2 = pd.read_csv('Data/dyn_chart_longer_cdiff.csv',sep=',')
p_services           = pd.read_csv('Data/services_pos.csv',sep=',')
p_drugs              = pd.read_csv('Data/drugs_pos.csv',sep=',',error_bad_lines=False)
p_microb             = pd.read_csv('Data/microb_pos.csv',sep=',')
n_static_features    = pd.read_csv('Data/info_negative.csv',sep=',')
n_dynamic_features   = pd.read_csv('Data/dyn_lab_longer_negative.csv',sep=',')
n_dynamic_features_2 = pd.read_csv('Data/dyn_chart_longer_negative.csv',sep=',')
n_services           = pd.read_csv('Data/services_neg.csv',sep=',')
n_drugs              = pd.read_csv('Data/drugs_neg.csv',sep=',',error_bad_lines=False)
n_microb             = pd.read_csv('Data/microb_neg.csv',sep=',')

static    = pd.concat([p_static_features,n_static_features])
dynamic_1 = pd.concat([p_dynamic_features,n_dynamic_features])
dynamic_2 = pd.concat([p_dynamic_features_2,n_dynamic_features_2])
services  = pd.concat([p_services,n_services])
drugs     = pd.concat([p_drugs,n_drugs])
microb    = pd.concat([p_microb,n_microb])

static.reset_index(drop=True, inplace=True)
dynamic_1.reset_index(drop=True, inplace=True)
dynamic_2.reset_index(drop=True, inplace=True)
services.reset_index(drop=True, inplace=True)
drugs.reset_index(drop=True, inplace=True)
microb.reset_index(drop=True, inplace=True)

'''
Loop to create indeces of output matrix
'''
indice_list = []    #Keep indices for each day
los_list    = []    #Length of stay
for i in range(0,static.shape[0]):
    patient = static['hadm_id'][i]
    admission_date = static['admittime'][i]
    discharge_date = static['dischtime'][i]
    label = static['label'][i]
    a = parser.parse(admission_date)
    b = parser.parse(discharge_date)
    lenght_of_stay = (b.date() - a.date()).days
    day_cur = 1
    for day in range(0,lenght_of_stay):
        #Create index
        indice_list.append(str(patient) + '_' + a.strftime('%Y-%m-%d'))
        a = a + timedelta(days=1)
        los_list.append(day_cur)
        day_cur += 1

out_indices = pd.Series(indice_list)
los_series  = pd.Series(los_list)


'''
Create empty matrix, this will be filled in later 
'''
output_matrix = pd.DataFrame(index=out_indices)
output_matrix = output_matrix.fillna(0) # with 0s rather than NaNs

output_matrix['temp_indices'] = output_matrix.index.astype(str)
output_matrix['hadm_id']      = output_matrix.temp_indices.str[:6]
output_matrix['los']          = pd.Series(los_list, index=output_matrix.index)
output_matrix.drop('temp_indices', axis=1, inplace=True)




'''
Start looping through the dynamic_features file
'''
for i in range(0,dynamic_1.shape[0]):
    cansu   = str(dynamic_1['hadm_id'][i]) + '_' + dynamic_1['date'][i]
    itemid  = dynamic_1['itemid'][i]
    output_matrix.set_value(cansu, itemid, dynamic_1['value'][i])

'''
Start looping through the dynamic_features_2 file
'''
for i in range(0,dynamic_2.shape[0]):
    cansu   = str(dynamic_2['hadm_id'][i]) + '_' + dynamic_2['date'][i]
    itemid  = dynamic_2['itemid'][i]
    if (itemid == 220045):
        itemid = 211
    elif (itemid == 220277):
        itemid = 646
    elif (itemid == 220210):
        itemid = 618
    elif (itemid == 220050):
        itemid = 51
    elif (itemid == 220051):
        itemid = 8368
    elif (itemid == 220052):
        itemid = 52
    elif (itemid == 220047):
        itemid = 5815
    elif (itemid == 220046):
        itemid = 8549
    elif (itemid == 223770):
        itemid = 5820
    elif (itemid == 223769):
        itemid = 8554
    elif (itemid == 224162):
        itemid = 5819
    elif (itemid == 224161):
        itemid = 8553
    elif (itemid == 220179):
        itemid = 455
    elif (itemid == 220180):
        itemid = 8441
    elif (itemid == 220181):
        itemid = 456
    elif (itemid == 223752):
        itemid = 5817
    elif (itemid == 223751):
        itemid = 8551
    elif (itemid == 220074):
        itemid = 113
    elif (itemid == 220056):
        itemid = 5813
    elif (itemid == 220058):
        itemid = 8547
    elif (itemid == 220739):
        itemid = 184
    elif (itemid == 223900):
        itemid = 723
    elif (itemid == 223901):
        itemid = 454
    elif (itemid == 223761 or itemid ==679):
        itemid = 678
    output_matrix.set_value(cansu, itemid, dynamic_2['value'][i])


'''
Start looping through the services file
'''
output_matrix['NSURG'] = 0
output_matrix['TSURG'] = 0
output_matrix['OBS'] = 0
output_matrix['MED'] = 0
output_matrix['GU'] = 0
output_matrix['CMED'] = 0
output_matrix['DENT'] = 0
output_matrix['NB'] = 0
output_matrix['GYN'] = 0
output_matrix['TRAUM'] = 0
output_matrix['ENT'] = 0
output_matrix['ORTHO'] = 0
output_matrix['OMED'] = 0
output_matrix['PSURG'] = 0
output_matrix['NBB'] = 0
output_matrix['SURG'] = 0
output_matrix['NMED'] = 0
output_matrix['CSURG'] = 0
output_matrix['VSURG'] = 0
output_matrix['PSYCH'] = 0

static_features = static.set_index(['hadm_id'])
for i in range(0,services.shape[0]):
    patient    = services['hadm_id'][i]
    start_date = services['transfertime'][i]
    disch_date = static_features.loc[patient]['dischtime']
    a = parser.parse(start_date)
    b = parser.parse(disch_date)
    lenght_of_stay = (b.date() - a.date()).days
    for day in range(0,lenght_of_stay):
        cansu   = str(patient) + '_' + a.strftime('%Y-%m-%d')
        itemid  = services['curr_service'][i]
        output_matrix.set_value(cansu, itemid, '1')
        a = a + timedelta(days=1)       

        
'''
Loop through the drugs file
'''        
with open('Data/DrugLists/AntibioticListFinal.txt') as f:
    antibiotic_list = f.read().splitlines() 

with open('Data/DrugLists/HighRiskAntibioticList.txt') as f:
    high_risk_antibiotic_list = f.read().splitlines() 

with open('Data/DrugLists/H2.txt') as f:
    h2_list = f.read().splitlines() 

with open('Data/DrugLists/ppi.txt') as f:
    ppi_list = f.read().splitlines() 

antibiotic_list = [x.lower() for x in antibiotic_list]
high_risk_antibiotic_list = [x.lower() for x in high_risk_antibiotic_list]
h2_list = [x.lower() for x in h2_list]
ppi_list = [x.lower() for x in ppi_list]

output_matrix['high_antib'] = 0
output_matrix['antib'] = 0
output_matrix['h2'] = 0
output_matrix['ppi'] = 0

#If no enddate, fill it with startdate. If no startdate, drop the line   
drugs.enddate.fillna(drugs.startdate, inplace=True)  
drugs.dropna(axis=0, inplace=True)
drugs.reset_index(drop=True, inplace=True)

for i in range(0,drugs.shape[0]):
    #print('i: ',i)
    #print('startdate:', drugs['startdate'][i])
    #print('enddate:', drugs['enddate'][i])
    patient    = drugs['hadm_id'][i]
    start_date = drugs['startdate'][i]
    end_date   = drugs['enddate'][i]
    admit_date = static_features.loc[patient]['admittime']
    disch_date = static_features.loc[patient]['dischtime']
    a = parser.parse(start_date)
    b = parser.parse(end_date)
    c = parser.parse(admit_date)
    d = parser.parse(disch_date)
    if ((a.date()>=c.date()) & (a.date()<=d.date())):
        lenght_of_stay = (b.date() - a.date()).days
        for day in range(0,lenght_of_stay):
            cansu      = str(patient) + '_' + a.strftime('%Y-%m-%d')
            drug_name  = drugs['drug'][i].lower()

            word = next((word for word in high_risk_antibiotic_list if word in drug_name), None)
            if word is not None:
                output_matrix.set_value(cansu, 'high_antib', '1')
                output_matrix.set_value(cansu, 'antib', '1')
            else:
                word = next((word for word in antibiotic_list if word in drug_name), None)
                if word is not None:
                    output_matrix.set_value(cansu, 'antib', '1')
            
            word = next((word for word in h2_list if word in drug_name), None)
            if word is not None:
                output_matrix.set_value(cansu, 'h2', '1')
            
            word = next((word for word in ppi_list if word in drug_name), None)
            if word is not None:
                output_matrix.set_value(cansu, 'ppi', '1')

            a = a + timedelta(days=1) 

'''
Loop through microbioloyevents data
'''
output_matrix['YEAST'] = 0
output_matrix['STAPHYLOCOCCUS, COAGULASE NEGATIVE'] = 0
output_matrix['STAPH AUREUS COAG +'] = 0
output_matrix['GRAM NEGATIVE ROD(S)'] = 0
output_matrix['PSEUDOMONAS AERUGINOSA'] = 0
output_matrix['KLEBSIELLA PNEUMONIAE'] = 0
output_matrix['ENTEROCOCCUS SP.'] = 0
output_matrix['ESCHERICHIA COLI'] = 0
output_matrix['PROTEUS MIRABILIS'] = 0
output_matrix['ENTEROBACTER CLOACAE'] = 0
output_matrix['KLEBSIELLA OXYTOCA'] = 0
output_matrix['ACINETOBACTER BAUMANNII'] = 0
output_matrix['SERRATIA MARCESCENS'] = 0
output_matrix['ENTEROBACTER AEROGENES'] = 0

for i in range(0,microb.shape[0]):
    cansu   = str(microb['hadm_id'][i]) + '_' + microb['date'][i]
    itemid  = microb['org_name'][i]
    output_matrix.set_value(cansu, itemid, microb['count'][i])
  
            
'''
Finalize and save the output file
'''
#Remove idendity columns, leave only data
output_matrix = output_matrix.reset_index()
output_matrix.rename(columns={'index': 'day_id'}, inplace=True)
#output_matrix.drop('hadm_id', axis=1, inplace=True)



#save the resulting file
output_matrix.to_csv('output_dynamics.csv', index=False)

'''
End
'''
print ('End time: ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

