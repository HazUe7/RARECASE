#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 0.1 Install necessary dependencies
get_ipython().system('pip install google-cloud-aiplatform --upgrade')


# In[3]:


# 0.2 Import required libraries
import pandas as pd
import numpy as np
import time
import os
import vertexai
import json
import re
from pydantic import BaseModel
from google import genai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part
from sklearn.preprocessing import LabelEncoder


# In[5]:


# 0.3 Set path variables
CSV_path = 'Cases per paper.csv'
input_dir = 'preprocess_outputs'
output_dir = 'LLM_extracted_IPD'

pilot_path = os.path.join(input_dir, "pilot_set.json")
development_path = os.path.join(input_dir, "development_set.json")
validation_path = os.path.join(input_dir, "validation_set.json")


# In[7]:


# 0.4 Load the splited datasets from JSON files
with open(pilot_path, "r", encoding="utf-8") as f:
    pilot_data = json.load(f)
    
with open(development_path, "r", encoding="utf-8") as f:
    development_data = json.load(f)
    
with open(validation_path, "r", encoding="utf-8") as f:
    validation_data = json.load(f)


# In[9]:


# 0.5 Load the cases-per-study CSV file and standardize the study name
cases_df = pd.read_csv(CSV_path)
cases_df["Study Name"] = cases_df["0"].astype(str)


# In[11]:


# 1.1 Initialize Vertex AI
from google.auth import default
from google.cloud import aiplatform

credentials, _ = default()
aiplatform.init(project="rarecase", location="us-central1")


# In[13]:


# 1.2 Set the generative model and define the prompt for code testing
model = GenerativeModel("gemini-2.0-flash-lite")

prompt = """
You are a medical researcher, processing individual patient data from a publication database on N-Methyl-D-Aspartate Receptor Antibody Encephalitis (NMDAR) for an Individual Patient Data Meta-Analysis (IPDMA).

Please identify NMDAR patients in the input publications and extract their following features:
- sex
- age (years)
- speech dysfunction (pressured speech, verbal reduction, mutism)
- seizures
- movement disorder, dyskinesias, or rigidity/abnormal postures
- decreased level of consciousness
- autonomic dysfunction or central hypoventilation
- abnormal (psychiatric) behaviour or cognitive dysfunction
- intensive care unit
- worst mRS in the acute phase
- EVER Focal or diff use slow or disorganised activity in EEG
- EVER Delta Brush in EEG
- presentation of epileptiform discharges in EEG
- CSF: Whether pleocytosis (Number of white blood cells per mm3 >= 5) or not?
- CSF: Number of white blood cells per mm3 (when pleocytosis)
- brain MRI Normal/Abnormal (ever)
- tumour
- days between onset and first immune therapy
- Whether the time between onset and first immune therapy ≤30 days?
- 1st line IT combination at 1st event (CS-IVIG-PE regardless of the order): <null, "None", "CS only", "CS+IVIG", "CS+IVIG+PE/IA", "CS+PE/IA", "IVIG only", "IVIG+PE/IA", "PE/IA only">
- RTX
- CYC
- Bortezomib
- Tocilizumab
- MMF for any duration
- AZA for any duration
- MTX for any duration
- Long-term steroids for ≥6 months
- Long-term IVIG for ≥6 months
- mRS at last follow-up

Output the result in JSON format. Each patient should be a dictionary with these fields.Binary features should be represented as 1 or 0, continuous features as numbers, and other features as strings. If the feature is unknown or not available or information is incomplete, leave it as null.

Output should be in the following format:
[
  {
    "Female": <Binary: Male encoded as 0, Female encoded as 1>,
    "Age": <Numeric>,
    "SpeechDys": <Binary: Yes encoded as 1, No encoded as 0>,
    "Seizures": <Binary: Yes encoded as 1, No encoded as 0>,
    "MvmtDis": <Binary: Yes encoded as 1, No encoded as 0>,
    "Obtunded": <Binary: Yes encoded as 1, No encoded as 0>,
    "BrainstemDys": <Binary: Yes encoded as 1, No encoded as 0>,
    "BehCogImp": <Binary: Yes encoded as 1, No encoded as 0>,
    "ITU": <Binary: Yes encoded as 1, No encoded as 0>,
    "WorstMRSAcute": <Numeric>,
    "EEGSlow": <Binary: Yes encoded as 1, No encoded as 0>,
    "EEGDeltaBrush": <Binary: Yes encoded as 1, No encoded as 0>,
    "EEGEpileptiform": <Binary: Yes encoded as 1, No encoded as 0>,
    "CSFPleo": <Binary: Yes encoded as 1, No encoded as 0>,
    "CSFWBCQuant": <Numeric>,
    "AbnMRIBrain": <Binary: Abnormal encoded as 1, Normal encoded as 0>,
    "Tumour": <Binary: Yes encoded as 1, No encoded as 0>,
    "DaysToIT": <Numeric>,
    "ITWithin30Days": <Binary: Yes encoded as 1, No encoded as 0>,
    "IT1stLineCombo": <Categorical: null, "None", "CS only", "CS+IVIG", "CS+IVIG+PE/IA", "CS+PE/IA", "IVIG only", "IVIG+PE/IA", "PE/IA only">,
    "IT2ndLineRTX": <Binary: Yes encoded as 1, No encoded as 0>,
    "IT2ndLineCYC": <Binary: Yes encoded as 1, No encoded as 0>,
    "IT2ndLineBort": <Binary: Yes encoded as 1, No encoded as 0>,
    "IT2ndLineToc": <Binary: Yes encoded as 1, No encoded as 0>,
    "ITMaintenanceMMF": <Binary: Yes encoded as 1, No encoded as 0>,
    "ITMaintenanceAZA": <Binary: Yes encoded as 1, No encoded as 0>,
    "ITMaintenanceMTX": <Binary: Yes encoded as 1, No encoded as 0>,
    "IT6mSteroid": <Binary: Yes encoded as 1, Not done for ≥6 months or not done at all encoded as 0.>,
    "IT6mIVIG": <Binary: Yes encoded as 1, Not done for ≥6 months or not done at all encoded as 0.>
    "FinalMRS": <Numeric>
  }
]
"""


# In[15]:


# 2.1 Define a function to extract and stucture IPD from single case reports 
def extract_case_report(model, prompt, study, text):
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
    
    response = model.generate_content(
        contents = [prompt, text],
        # config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)),  ### To disable thinking if using gemini-2.5
        generation_config = GenerationConfig(temperature=0),
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
    FinalMRS = IPD.get("FinalMRS", None)
    
    output = [
        study, duration,
        Female, Age, SpeechDys, Seizures, MvmtDis, Obtunded, BrainstemDys, BehCogImp, ITU, WorstMRSAcute,
        EEGSlow, EEGDeltaBrush, EEGEpileptiform, CSFPleo, CSFWBCQuant, AbnMRIBrain, Tumour, DaysToIT, ITWithin30Days, IT1stLineCombo,
        IT2ndLineRTX, IT2ndLineCYC, IT2ndLineBort, IT2ndLineToc, ITMaintenanceMMF, ITMaintenanceAZA, ITMaintenanceMTX, IT6mSteroid, IT6mIVIG, FinalMRS
    ]
    
    return output


# In[17]:


# 2.2 Define a function to extract and stucture multiple entries of IPD from case series 
def extract_case_series(model, prompt, study, text):
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
    
    response = model.generate_content(
        contents = [prompt, text],
        # config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)),  ### To disable thinking if using gemini-2.5
        generation_config = GenerationConfig(temperature=0),
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
        FinalMRS = IPD.get("FinalMRS", None)
    
        records = [
            study, duration,
            Female, Age, SpeechDys, Seizures, MvmtDis, Obtunded, BrainstemDys, BehCogImp, ITU, WorstMRSAcute,
            EEGSlow, EEGDeltaBrush, EEGEpileptiform, CSFPleo, CSFWBCQuant, AbnMRIBrain, Tumour, DaysToIT, ITWithin30Days, IT1stLineCombo,
            IT2ndLineRTX, IT2ndLineCYC, IT2ndLineBort, IT2ndLineToc, ITMaintenanceMMF, ITMaintenanceAZA, ITMaintenanceMTX, IT6mSteroid, IT6mIVIG, FinalMRS
        ]
        output.append(records)
        
    return output


# In[19]:


# 2.3 Define a function to batch process a dataset of study and texts
def batch_feature_extraction(model, prompt, dataset):
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
            records.append(extract_case_report(model, prompt, study, text))
        else:
            series_records = extract_case_series(model, prompt, study, text)
            records.extend(series_records)
            
        time.sleep(5)                                                                       ### Pause 5 seconds between each request to avoid rate limiting.

    df = pd.DataFrame(records, columns=["Study", "Duration", 
                                        "Female", "Age", "SpeechDys", "Seizures", "MvmtDis", "Obtunded", "BrainstemDys", "BehCogImp", "ITU", "WorstMRSAcute",
                                        "EEGSlow", "EEGDeltaBrush", "EEGEpileptiform", "CSFPleo", "CSFWBCQuant", "AbnMRIBrain", "Tumour", "DaysToIT", "ITWithin30Days", "IT1stLineCombo",
                                        "IT2ndLineRTX", "IT2ndLineCYC", "IT2ndLineBort", "IT2ndLineToc", "ITMaintenanceMMF", "ITMaintenanceAZA", "ITMaintenanceMTX", "IT6mSteroid", "IT6mIVIG", "FinalMRS"
                                       ])
    return df                                                                               ### Convert all records into a DataFrame


# In[21]:


# 3.1 Run batch extraction on the pilot dataset
pilot_output = batch_feature_extraction(model, prompt, pilot_data)
print(pilot_output)

pilot_output_path = os.path.join(output_dir, 'pilot_output.csv')
pilot_output.to_csv(pilot_output_path, index=False)                                    ### Save the pilot output as a CSV file


# In[23]:


# 3.2 Run batch extraction on the development dataset
development_output = batch_feature_extraction(model, prompt, development_data)
print(development_output.head())

development_output_path = os.path.join(output_dir, 'development_output.csv')
development_output.to_csv(development_output_path, index=False)                             ### Save the development output as a CSV file

