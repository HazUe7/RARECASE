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


# In[7]:


# 0.3 Set path variables
CSV_path = 'Cases per paper.csv'
input_dir = 'preprocess_outputs'
output_dir = 'LLM_extracted_IPD'

pilot_path = os.path.join(input_dir, "pilot_set.json")
development_path = os.path.join(input_dir, "development_set.json")
validation_path = os.path.join(input_dir, "validation_set.json")


# In[9]:


# 0.4 Load the splited datasets from JSON files
with open(pilot_path, "r", encoding="utf-8") as f:
    pilot_data = json.load(f)
    
with open(development_path, "r", encoding="utf-8") as f:
    development_data = json.load(f)
    
with open(validation_path, "r", encoding="utf-8") as f:
    validation_data = json.load(f)


# In[11]:


# 0.5 Load the cases-per-study CSV file and standardize the study name
cases_df = pd.read_csv(CSV_path)
cases_df["Study Name"] = cases_df["0"].astype(str)


# In[13]:


# 1.1 Initialize Vertex AI
from google.auth import default
from google.cloud import aiplatform

credentials, _ = default()
aiplatform.init(project="rarecase", location="us-central1")


# In[15]:


# 1.2 Set the generative model and define the prompt for code testing
model = GenerativeModel("gemini-2.0-flash-lite")

prompt = """
You are a medical researcher, processing individual patient data from a publication database on N-Methyl-D-Aspartate Receptor Antibody Encephalitis (NMDAR) for an Individual Patient Data Meta-Analysis (IPDMA).
Please identify NMDAR patients in the input publications and extract their following features:
- age (years)
- sex
- tumor
- number of main group of symptoms
- length of hospital stay (days)
- T2 and FLAIR hyperintensities
Output the result in JSON format. Each patient should be a dictionary with these fields.Binary features should be represented as 1 or 0, continuous features as numbers, and other features as strings. If the feature is unknown or not available, leave it as blank.
Output should be in the following format:
[
  {
    "age (years)": <Numeric>,
    "sex": <Binary: Male encoded as 0, Female encoded as 1>,
    "tumor": <Binary: Yes encoded as 1, No encoded as 0>,
    “number of main group of symptoms”: <Numeric>,
    "length of hospital stay (days)": <Numeric>,
    "T2 and FLAIR hyperintensities": <Categorical: 'y', 'Normal', 'MRI n.a./not done', 'n', 'CT Normal', 'Abnormal', 'n.a.', 'Previous head trauma Abnormal'>,
  }
]
"""


# In[17]:


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
    
    print(output_json)
    print("\n" + "-"*50 + "\n")

    data = json.loads(output_json)                                                                      ### Parse the JSON string to Python data structures                   
    
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):                          ### Handle case where output is a list with a dict or just a dict
        IPD = data[0]
    elif isinstance(data, dict):
        IPD = data
        
    age = IPD.get("age (years)", None)
    sex = IPD.get("sex", None)
    tumor = IPD.get("tumor", None)
    n_main_symptoms = IPD.get("number of main group of symptoms", None)
    len_stay = IPD.get("length of hospital stay (days)", None)
    mri_t2_flair = IPD.get("T2 and FLAIR hyperintensities", None)

    output = [study, duration, age, sex, tumor, n_main_symptoms, len_stay, mri_t2_flair]
    
    return output


# In[19]:


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
    
    print(output_json)
    print("\n" + "-"*50 + "\n")

    data = json.loads(output_json)
    
    output = []                                                                     ### Define a list to contain lists of each series
    for IPD in data:                                                                ### Iterate over each patient's data in the series
        age = IPD.get("age (years)", None)
        sex = IPD.get("sex", None)
        tumor = IPD.get("tumor", None)
        n_main_symptoms = IPD.get("number of main group of symptoms", None)
        len_stay = IPD.get("length of hospital stay (days)", None)
        mri_t2_flair = IPD.get("T2 and FLAIR hyperintensities", None)
    
        records = [study, duration, age, sex, tumor, n_main_symptoms, len_stay, mri_t2_flair]
        output.append(records)
        
    return output


# In[21]:


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
            series_records = extract_case_series(model, prompt, text, study)
            records.extend(series_records)
            
        time.sleep(5)                                                                       ### Pause 5 seconds between each request to avoid rate limiting.

    df = pd.DataFrame(records, columns=["Study", "Duration", "Age (years)", "Sex", "Tumor", "Number of Main Group of Symptoms", "Length of Hospital Stay (days)", "T2 and FLAIR Hyperintensities"])
    return df                                                                               ### Convert all records into a DataFrame


# In[23]:


# 3.1 Run batch extraction on the pilot dataset
pilot_output = batch_feature_extraction(model, prompt, pilot_data)
print(pilot_output)

pilot_output_path = os.path.join(output_dir, 'pilot_output.csv')
pilot_output.to_csv(pilot_output_path, index=False)                                    ### Save the pilot output as a CSV file


# In[25]:


# 3.2 Run batch extraction on the development dataset
development_output = batch_feature_extraction(model, prompt, development_data)
print(development_output.head())

development_output_path = os.path.join(output_dir, 'development_output.csv')
development_output.to_csv(development_output_path, index=False)                             ### Save the development output as a CSV file

