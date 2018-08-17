import numpy as np
import pandas as pd

##############################################################################
### Cansu Sen
### Create medical history features
### Input: CSV files generated through MIMIC III database from the following tables:
###        Noteevents (Discharge Summaries)
### Output: Static medical history features for CDIF cohort
### MIMIC III is a publicly available critical care dataset. We are not at the
### liberty of sharing the exact data files we used, however, one can obtain 
### access to MIMIC dataset and create these files. 
### MIMIC III - https://mimic.physionet.org
##############################################################################


################################################################################
### READ DISCHARGE SUMMARIES FOR THE PATIENT SET WHICH INCLUDES POSITIVE AND NEGATIVE PATEINTS
################################################################################
df = pd.read_csv('Data/all_discharge_summaries.csv',sep='\\')
print('Number of notes: ',df.shape[0])


################################################################################
### EXTRACT MEDICATION ON ADMISSION PART AND SAVE IT
################################################################################
start_terms = ['Medications on Admission', 'MEDICATIONS ON ADMISSION', 'ADMISSION MEDICATIONS', 'Admission Medications','MEDICATIONS:  On admission',
              'Home MEDS','Home Medications','MEDICATIONS AT HOME','MEDICATIONS ON TRANSFER','HOME MEDICATIONS','Transfer Medications',
              'MEDICATIONS UPON TRANSFER','MEDICATIONS:  (on admission)','MEDICATIONS ON ARRIVAL','MEDICATIONS ON PRESENTATION',
              'MEDICATIONS PRIOR TO ADMISSION', 'Medications on admission',
              '\nMEDICATIONS:'] 
### BE CAREFUL ABOUT MEDICATIONS:

end_terms = ['Discharge Medications', 'ALLERGIES', 'SOCIAL HISTORY','DISCHARGE MEDICATIONS', 'Allergies', 
             'Social History', 'Discharge Disposition', 'Disposition','PHYSICAL EXAMINATION',
            'HOSPITAL COURSE','FAMILY HISTORY','Family History','VITAL SIGNS ON ADMISSION',
            'CAR SEAT', 'IMMUNIZATION', 'DISCHARGE', 'CAR-SEAT', 'OTHER', 'DIAGNOSIS']

output_hadm_id = []
output_med_text = []
err_cnt = 0

for i in range(0,df.shape[0]):
    note = df['text'][i]
    #note = note.lower()
    
    sentence = '--'
    start_indices = []
    end_indices = []
    start_index = -1
    end_index = -1

    for item in start_terms:
        temp_index = note.find(item)
        start_indices.append(temp_index)

    for item in end_terms:
        temp_index = note.find(item)
        end_indices.append(temp_index)
    
    if ([j for j in start_indices if j > 0]):
        start_index = min(j for j in start_indices if j > 0)
        if ([k for k in end_indices if k > start_index]):
            end_index = min(k for k in end_indices if k > start_index)
            sentence = note[start_index:end_index]
            output_hadm_id.append(df['hadm_id'][i])
            output_med_text.append(sentence)
        else:
            print(df['row_id'][i])
    else:
        err_cnt = err_cnt +1
        
tmp1    = pd.Series(output_hadm_id)
tmp2    = pd.Series(output_med_text)
column_list = ['hadm_id', 'medications'] 
output_med = pd.DataFrame(columns=column_list)
output_med['hadm_id'] = tmp1
output_med['medications'] = tmp2
output_med.to_csv('_medications_on_admission.csv', sep='\\', index=False)


################################################################################
### EXTRACT PAST MEDICAL HISTORY PART AND SAVE IT
################################################################################
start_terms = ['Past Medical History', 'PAST MEDICAL HISTORY']
#start_terms = ['Past Medical History', 'PAST MEDICAL HISTORY','PAST ILLNESSES','past medical history','PAST MEDICAL/SURGICAL HISTORY']
#PAST ILLNESSES, past medical history, PAST MEDICAL/SURGICAL HISTORY:
end_terms = ['Social History', 'SOCIAL HISTORY', 'Past Surgical History', 'Past surgical history','Past history', 
             'MEDICATIONS ON ADMISSION','PAST SURGICAL HISTORY','MEDICATIONS', 'HOSPITAL COURSE', 'ALLERGIES',
             'Physical Exam','Family History','Pertinent Results','Brief Hospital Course']

output_hadm_id = []
output_pmh_text = []
err_cnt = 0

for i in range(0,df.shape[0]):
    note = df['text'][i]
    #note = note.lower()
    
    sentence = '--'
    start_indices = []
    end_indices = []
    start_index = -1
    end_index = -1

    for item in start_terms:
        temp_index = note.find(item)
        start_indices.append(temp_index)

    for item in end_terms:
        temp_index = note.find(item)
        end_indices.append(temp_index)
    
    if ([j for j in start_indices if j > 0]):
        start_index = min(j for j in start_indices if j > 0)
        if ([k for k in end_indices if k > start_index]):
            end_index = min(k for k in end_indices if k > start_index)
            sentence = note[start_index:end_index]   
            output_hadm_id.append(df['hadm_id'][i])
            output_pmh_text.append(sentence)    
        else:
            print('Couldnt decide end index.', df['row_id'][i])
    else:
        err_cnt = err_cnt +1
        
#print(err_cnt)
tmp1    = pd.Series(output_hadm_id)
tmp2    = pd.Series(output_pmh_text)
column_list = ['hadm_id', 'pmh'] 
output_pmh = pd.DataFrame(columns=column_list)
output_pmh['hadm_id'] = tmp1
output_pmh['pmh'] = tmp2
output_pmh.to_csv('_past_medical_history.csv', sep='\\', index=False)


################################################################################
### SETTINGS FOR PREVIOUS's
################################################################################
df = pd.read_csv('_past_medical_history.csv',sep='\\')
patients = pd.read_csv('patient_list.csv',sep='\\')
print('How many notes: ',df.shape[0])
print('------Starting------ ')

out_feat = pd.DataFrame(index=patients['hadm_id'].values, columns=['kidney','diabetes','cdiff','antib'])
out_feat['kidney'] = 0 
out_feat['diabetes'] = 0 
out_feat['cdiff'] = 0 
out_feat['antib'] = 0 

################################################################################
### EXTRACT FEATURE FOR KIDNEY DISEASE
################################################################################
for i in range(0,df.shape[0]):
    curr_index = df['hadm_id'][i]
    
    if 'kidney' in df['pmh'][i].lower():
        out_feat['kidney'][curr_index] = 1
    elif 'dialysis'in df['pmh'][i].lower():
        out_feat['kidney'][curr_index] = 1
    elif ' renal 'in df['pmh'][i].lower():
        out_feat['kidney'][curr_index] = 1
            

################################################################################
### DIABETES
################################################################################

for i in range(0,df.shape[0]):
    curr_index = df['hadm_id'][i]

    #''' Version 2
    for item in df['pmh'][i].lower().split("\n"):
        if 'diabetes' in item:
            if 'cardiac risk factors' in item:
                if '-' in item:
                    out_feat['diabetes'][curr_index] = 0
                else:
                    out_feat['diabetes'][curr_index] = 1
            else:
                out_feat['diabetes'][curr_index] = 1

    
################################################################################
### CDIFF
################################################################################

for i in range(0,df.shape[0]):
    curr_index = df['hadm_id'][i]

    #''' Version 2
    for item in df['pmh'][i].lower().split("\n"):
        if 'diff ' in item:
            out_feat['cdiff'][curr_index] = 1
        elif 'difficil' in item:
            out_feat['cdiff'][curr_index] = 1
            
################################################################################
### ANTIBIOTIC
################################################################################
df = pd.read_csv('_medications_on_admission.csv',sep='\\')

fname = 'DrugLists/AntibioticListFinal.txt'
with open(fname) as f:
    antibiotic_list = f.read().splitlines() 
count = 0

for i in range(0,df.shape[0]):
    curr_index = df['hadm_id'][i]
    note = df['medications'][i]
    
    word = next((word for word in antibiotic_list if word in note), None)
    if word is not None:
        #print the word that fired the any() function
        #print('-----------')
        #print(word)
        out_feat['antib'][curr_index] = 1
        count +=1
    else:
        out_feat['antib'][curr_index] = 0
        
################################################################################
### SAVE
################################################################################
out_feat.to_csv('out_previous_features.csv', index='False')