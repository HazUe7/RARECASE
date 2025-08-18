#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 0.1 Install necessary dependencies
get_ipython().system('pip install -U google-genai')


# In[1]:


# 0.2 Import required libraries
import pandas as pd
import numpy as np
import time
import os
import json
import re
from google import genai
from google.genai import types


# In[3]:


# 0.3 Set path variables
CSV_path = 'Cases per paper.csv'
input_dir = 'preprocess_outputs'
output_dir = 'LLM_extracted_IPD'

pilot_path = os.path.join(input_dir, "pilot_set.json")
development_path = os.path.join(input_dir, "development_set.json")
validation_path = os.path.join(input_dir, "validation_set.json")


# In[5]:


# 0.4 Load the splited datasets from JSON files
with open(pilot_path, "r", encoding="utf-8") as f:
    pilot_data = json.load(f)
    
with open(development_path, "r", encoding="utf-8") as f:
    development_data = json.load(f)
    
with open(validation_path, "r", encoding="utf-8") as f:
    validation_data = json.load(f)


# In[7]:


# 0.5 Load the cases-per-study CSV file and standardize the study name
cases_df = pd.read_csv(CSV_path)
cases_df["Study Name"] = cases_df["0"].astype(str)


# In[9]:


# 1.1 Initialize Gemini AI client using API key from environment variable
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# In[11]:


# 1.2 Set the prompt for code testing
prompt = """
You are a medical researcher, processing individual patient data from a publication database on N-Methyl-D-Aspartate Receptor Antibody Encephalitis (NMDARE) for an Individual Patient Data Meta-Analysis (IPDMA).

Please identify NMDARE patients in the input publications. 
The inclusion criteria for cases are: (1) patients with NMDARE with positive NMDAR antibodies in serum and/or cerebrospinal fluid (CSF); (2) their individual patient data on immunotherapy was provided.
The exclusion criteria for cases are: (1) patients with NMDARE preceded by central nervous system infection (e.g., herpes simplex virus encephalitis); (2) patients in large cohorts where individual patient details were not available.

Instruction:
(1) Score clinical and treatment features as Yes if they occurred at any time during the first disease event (i.e. first clinical episode of NMDARE), disregarding any subsequent relapses. Extract continuous and categorical features values from the first event where applicable. However MonthsToFU, FinalMRS and Relapse regard the whole disease course.
(2) Binary features should be represented as 1 (Female/Abnormal/Yes/Present) or 0 (Male/Normal/No/Absent), continuous features as numbers, and categorical features as strings (choose the matching category from the provided list). If the feature is unknown, unavailable or incomplete, leave it as null.


Then extract the following features for each patient. 
- Female: sex (female or not)
- Age: age at disease onset (years)
- SpeechDys: presence of speech dysfunction (e.g. pressured speech, verbal reduction, mutism)
- Seizures: presence of seizures
- MvmtDis: presence of movement disorder (e.g. dyskinesias, or rigidity/abnormal postures)
- Obtunded: presence of decreased level of consciousness
- BrainstemDys: presence of autonomic dysfunction or central hypoventilation
- BehCogImp: presence of abnormal (psychiatric) behaviour or cognitive dysfunction
- ITU: whether admitted to an intensive care unit
- WorstMRSAcute: worst mRS (modified Rankin Scale (integer score ranging from 0 (no symptoms) to 6 (dead)) for assessing functional independence and disability in patients with neurological disorders) in the acute phase
- EEGSlow: presence of focal or diffuse slow or disorganised activity in EEG (Electroencephalography)
- EEGDeltaBrush: presence of delta brush in EEG
- EEGEpileptiform: presentation of epileptiform discharges in EEG
- CSFPleo: whether pleocytosis in CSF(Cerebrospinal Fluid) (i.e. Number of white blood cells per mm3 >= 5) 
- CSFWBCQuant: maximum number of white blood cells per mm3 in CSF 
- AbnMRIBrain: brain MRI (Magnetic Resonance Imaging) abnormal (increased T2/FLAIR parenchymal signal intensity or contrast enhancement)
- Tumour: presence of tumour
- DaysToIT: number of days between symptom onset and first immunotherapy (days)
- ITWithin30Days: whether the time between onset and first immunotherapy ≤30 days?
- IT1stLineCombo: the specific combination of first line immunotherapies the patient received at their first disease event. (First-line immunotherapies include: corticosteroids [CS], Intravenous Immunoglobulin [IVIG], therapeutic plasma exchange [PE]/immunoadsorption [IA]. Matched the drug combination to one of the following categories, regardless of the order: "None", "CS only", "CS+IVIG", "CS+IVIG+PE/IA", "CS+PE/IA", "IVIG only", "IVIG+PE/IA", "PE/IA only")
- IT2ndLineRTX: initiation of Rituximab (RTX) at first disease event
- IT2ndLineCYC: initiation of Cyclophosphamide (CYC) at first disease event
- IT2ndLineBort: initiation of Bortezomib at first disease event
- IT2ndLineToc: initiation of Tocilizumab at first disease event
- ITMaintenanceMMF: initiation of Mycophenolate Mofetil (MMF) for any duration at first disease event
- ITMaintenanceAZA: initiation of Azathioprine (AZA) for any duration at first disease event
- ITMaintenanceMTX: initiation of Methotrexate (MTX) for any duration at first disease event
- IT6mSteroid: initiation of long-term steroids for ≥6 months at first disease event
- IT6mIVIG: initiation of long-term IVIG use for ≥6 months at first disease event
- MonthsToFU: number of months from onset of NMDARE to last follow-up (months)
- FinalMRS: mRS at last follow-up
- Relapse: presence of relapses of NMDARE

Output the result in JSON format, where each patient should be a dictionary with these fields. Only output the JSON list and do not include any extra text or explanation.
Output should be in the following format:
[
  {
    "Female": <Binary: Female encoded as 1, Male encoded as 0>,
    "Age": <Numeric: float>,
    "SpeechDys": <Binary: Yes encoded as 1, No encoded as 0>,
    "Seizures": <Binary: Yes encoded as 1, No encoded as 0>,
    "MvmtDis": <Binary: Yes encoded as 1, No encoded as 0>,
    "Obtunded": <Binary: Yes encoded as 1, No encoded as 0>,
    "BrainstemDys": <Binary: Yes encoded as 1, No encoded as 0>,
    "BehCogImp": <Binary: Yes encoded as 1, No encoded as 0>,
    "ITU": <Binary: Yes encoded as 1, No encoded as 0>,
    "WorstMRSAcute": <Numeric: integer (0-6)>,
    "EEGSlow": <Binary: Yes encoded as 1, No encoded as 0>,
    "EEGDeltaBrush": <Binary: Yes encoded as 1, No encoded as 0>,
    "EEGEpileptiform": <Binary: Yes encoded as 1, No encoded as 0>,
    "CSFPleo": <Binary: Yes encoded as 1, No encoded as 0>,
    "CSFWBCQuant": <Numeric: integer>,
    "AbnMRIBrain": <Binary: Abnormal encoded as 1, Normal encoded as 0>,
    "Tumour": <Binary: Yes encoded as 1, No encoded as 0>,
    "DaysToIT": <Numeric: integer>,
    "ITWithin30Days": <Binary: Yes encoded as 1, No encoded as 0>,
    "IT1stLineCombo": <Categorical: "None", "CS only", "CS+IVIG", "CS+IVIG+PE/IA", "CS+PE/IA", "IVIG only", "IVIG+PE/IA", "PE/IA only">,
    "IT2ndLineRTX": <Binary: Yes encoded as 1, No encoded as 0>,
    "IT2ndLineCYC": <Binary: Yes encoded as 1, No encoded as 0>,
    "IT2ndLineBort": <Binary: Yes encoded as 1, No encoded as 0>,
    "IT2ndLineToc": <Binary: Yes encoded as 1, No encoded as 0>,
    "ITMaintenanceMMF": <Binary: Yes encoded as 1, No encoded as 0>,
    "ITMaintenanceAZA": <Binary: Yes encoded as 1, No encoded as 0>,
    "ITMaintenanceMTX": <Binary: Yes encoded as 1, No encoded as 0>,
    "IT6mSteroid": <Binary: Yes encoded as 1, Not done for ≥6 months or not done at all encoded as 0.>,
    "IT6mIVIG": <Binary: Yes encoded as 1, Not done for ≥6 months or not done at all encoded as 0.>,
    "MonthsToFU": <Numeric: float>,
    "FinalMRS": <Numeric: integer (0-6)>,
    "Relapse": <Binary: Yes encoded as 1, No encoded as 0>
  }
]
"""


# In[13]:


# 2.1 Define a function to extract and stucture IPD from single case reports 
def extract_case_report(prompt, study, text):
    """
    Extract IPD from a single case report.
    Parameters:
        model: The LLM instance used.
        prompt: The prompt text guiding extraction.
        study: Identifier of the study.
        text: The full text content of the single case report.
    Returns:
        A list containing the study name, extraction duration, and extracted patient features.
    """
    start_time = time.time()                                                                            ### Record start/end time for measuring duration

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, text],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0
        )
    )
    
    end_time = time.time()
    duration = end_time - start_time

    output_text = response.candidates[0].content.parts[0].text                                          ### Extract the text output from the model's response
    output_json = output_text.strip().removeprefix("```json").removesuffix("```").strip()               ### Clean up JSON code block markers

    data = json.loads(output_json)                                                                      ### Parse the JSON string to Python data structures                   
    
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):                          ### Handle case where output is a list with a dict or just a dict
        IPD = data[0]
    elif isinstance(data, dict):
        IPD = data
        
    Female = IPD.get("Female", None)
    Age = IPD.get("Age", None)
    SpeechDys = IPD.get("SpeechDys", None)
    Seizures = IPD.get("Seizures", None)
    MvmtDis = IPD.get("MvmtDis", None)
    Obtunded = IPD.get("Obtunded", None)
    BrainstemDys = IPD.get("BrainstemDys", None)
    BehCogImp = IPD.get("BehCogImp", None)
    ITU = IPD.get("ITU", None)
    WorstMRSAcute = IPD.get("WorstMRSAcute", None)
    EEGSlow = IPD.get("EEGSlow", None)
    EEGDeltaBrush = IPD.get("EEGDeltaBrush", None)
    EEGEpileptiform = IPD.get("EEGEpileptiform", None)
    CSFPleo = IPD.get("CSFPleo", None)
    CSFWBCQuant = IPD.get("CSFWBCQuant", None)
    AbnMRIBrain = IPD.get("ITU", None)
    Tumour = IPD.get("Tumour", None)
    DaysToIT = IPD.get("DaysToIT", None)
    ITWithin30Days = IPD.get("ITWithin30Days", None)
    IT1stLineCombo = IPD.get("IT1stLineCombo", None)
    IT2ndLineRTX = IPD.get("IT2ndLineRTX", None)
    IT2ndLineCYC = IPD.get("IT2ndLineCYC", None)
    IT2ndLineBort = IPD.get("IT2ndLineBort", None)
    IT2ndLineToc = IPD.get("IT2ndLineToc", None)
    ITMaintenanceMMF = IPD.get("ITMaintenanceMMF", None)
    ITMaintenanceAZA = IPD.get("ITMaintenanceAZA", None)
    ITMaintenanceMTX = IPD.get("ITMaintenanceMTX", None)
    IT6mSteroid = IPD.get("IT6mSteroid", None)
    IT6mIVIG = IPD.get("IT6mIVIG", None)
    MonthsToFU = IPD.get("MonthsToFU", None)
    FinalMRS = IPD.get("FinalMRS", None)
    Relapse = IPD.get("Relapse", None)
    
    output = [
        study, duration,
        Female, Age, SpeechDys, Seizures, MvmtDis, Obtunded, BrainstemDys, BehCogImp, ITU, WorstMRSAcute,
        EEGSlow, EEGDeltaBrush, EEGEpileptiform, CSFPleo, CSFWBCQuant, AbnMRIBrain, Tumour, DaysToIT, ITWithin30Days, IT1stLineCombo,
        IT2ndLineRTX, IT2ndLineCYC, IT2ndLineBort, IT2ndLineToc, ITMaintenanceMMF, ITMaintenanceAZA, ITMaintenanceMTX, IT6mSteroid, IT6mIVIG, MonthsToFU,
        FinalMRS, Relapse
    ]
    
    return output


# In[15]:


# 2.2 Define a function to extract and stucture multiple entries of IPD from case series 
def extract_case_series(prompt, study, text):
    """
    Extract IPD from a case series.
    Parameters:
        model: The LLM instance used.
        prompt: The prompt text guiding extraction.
        study: Identifier of the study.
        text: The full text content of the single case report.
    Returns:
        A list of lists, each inner list representing extracted features for a single patient in the series.
    """
    start_time = time.time()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, text],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0
        )
    )
    
    end_time = time.time()
    duration = end_time - start_time

    output_text = response.candidates[0].content.parts[0].text
    output_json = output_text.strip().removeprefix("```json").removesuffix("```").strip()

    data = json.loads(output_json)
    
    output = []                                                                     ### Define a list to contain lists of each series
    for IPD in data:                                                                ### Iterate over each patient's data in the series
        Female = IPD.get("Female", None)
        Age = IPD.get("Age", None)
        SpeechDys = IPD.get("SpeechDys", None)
        Seizures = IPD.get("Seizures", None)
        MvmtDis = IPD.get("MvmtDis", None)
        Obtunded = IPD.get("Obtunded", None)
        BrainstemDys = IPD.get("BrainstemDys", None)
        BehCogImp = IPD.get("BehCogImp", None)
        ITU = IPD.get("ITU", None)
        WorstMRSAcute = IPD.get("WorstMRSAcute", None)
        EEGSlow = IPD.get("EEGSlow", None)
        EEGDeltaBrush = IPD.get("EEGDeltaBrush", None)
        EEGEpileptiform = IPD.get("EEGEpileptiform", None)
        CSFPleo = IPD.get("CSFPleo", None)
        CSFWBCQuant = IPD.get("CSFWBCQuant", None)
        AbnMRIBrain = IPD.get("ITU", None)
        Tumour = IPD.get("Tumour", None)
        DaysToIT = IPD.get("DaysToIT", None)
        ITWithin30Days = IPD.get("ITWithin30Days", None)
        IT1stLineCombo = IPD.get("IT1stLineCombo", None)
        IT2ndLineRTX = IPD.get("IT2ndLineRTX", None)
        IT2ndLineCYC = IPD.get("IT2ndLineCYC", None)
        IT2ndLineBort = IPD.get("IT2ndLineBort", None)
        IT2ndLineToc = IPD.get("IT2ndLineToc", None)
        ITMaintenanceMMF = IPD.get("ITMaintenanceMMF", None)
        ITMaintenanceAZA = IPD.get("ITMaintenanceAZA", None)
        ITMaintenanceMTX = IPD.get("ITMaintenanceMTX", None)
        IT6mSteroid = IPD.get("IT6mSteroid", None)
        IT6mIVIG = IPD.get("IT6mIVIG", None)
        MonthsToFU = IPD.get("MonthsToFU", None)
        FinalMRS = IPD.get("FinalMRS", None)
        Relapse = IPD.get("Relapse", None)
    
        records = [
            study, duration,
            Female, Age, SpeechDys, Seizures, MvmtDis, Obtunded, BrainstemDys, BehCogImp, ITU, WorstMRSAcute,
            EEGSlow, EEGDeltaBrush, EEGEpileptiform, CSFPleo, CSFWBCQuant, AbnMRIBrain, Tumour, DaysToIT, ITWithin30Days, IT1stLineCombo,
            IT2ndLineRTX, IT2ndLineCYC, IT2ndLineBort, IT2ndLineToc, ITMaintenanceMMF, ITMaintenanceAZA, ITMaintenanceMTX, IT6mSteroid, IT6mIVIG, MonthsToFU,
            FinalMRS, Relapse
        ]
        output.append(records)
        
    return output


# In[17]:


# 2.3 Define a function to batch process a dataset of study and texts
def batch_feature_extraction(prompt, dataset):
    """
    Extract IPD either as single case reports or case series depending on the number of cases.
    Parameters:
        model: The LLM instance used.
        prompt: The prompt text guiding extraction.
        dataset: Dictionary mapping study identifiers to their full text content.
    Returns:
        A pandas DataFrame containing all extracted features for all studies/cases.
    """
    records = []
    for study, text in dataset.items():
        n_cases = cases_df.loc[cases_df["Study Name"] == study, "count"].values[0]          ### Look up number of cases for this study
        
        if n_cases == 1:
            records.append(extract_case_report(prompt, study, text))
        else:
            series_records = extract_case_series(prompt, study, text)
            records.extend(series_records)
            
        time.sleep(5)                                                                       ### Pause 5 seconds between each request to avoid rate limiting.

    df = pd.DataFrame(records, columns=["Study", "Duration", 
                                        "Female", "Age", "SpeechDys", "Seizures", "MvmtDis", "Obtunded", "BrainstemDys", "BehCogImp", "ITU", "WorstMRSAcute",
                                        "EEGSlow", "EEGDeltaBrush", "EEGEpileptiform", "CSFPleo", "CSFWBCQuant", "AbnMRIBrain", "Tumour", "DaysToIT", "ITWithin30Days", "IT1stLineCombo",
                                        "IT2ndLineRTX", "IT2ndLineCYC", "IT2ndLineBort", "IT2ndLineToc", "ITMaintenanceMMF", "ITMaintenanceAZA", "ITMaintenanceMTX", "IT6mSteroid", "IT6mIVIG", "MonthsToFU",
                                        "FinalMRS", "Relapse"
                                       ])
    return df                                                                               ### Convert all records into a DataFrame


# In[19]:


# 3.1 Run batch extraction on the pilot dataset
pilot_output = batch_feature_extraction(prompt, pilot_data)
print(pilot_output)

pilot_output_path = os.path.join(output_dir, 'pilot_output.csv')
pilot_output.to_csv(pilot_output_path, index=False)                                    ### Save the pilot output as a CSV file


# In[21]:


# 3.2 Run batch extraction on the development dataset
development_output = batch_feature_extraction(prompt, development_data)
print(development_output.head())

development_output_path = os.path.join(output_dir, 'development_output.csv')
development_output.to_csv(development_output_path, index=False)                             ### Save the development output as a CSV file

