#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 0.1 Import required libraries
import pandas as pd
import numpy as np
import time
import re
import os
import replicate
from replicate.client import Client


# In[3]:


# 0.2 Set path variables
CSV_path = 'Cases per paper.csv'
input_dir = 'preprocess_outputs'
output_dir = 'LLM_extracted_IPD'

pilot_path = os.path.join(input_dir, "pilot_set.json")
development_path = os.path.join(input_dir, "development_set.json")
validation_path = os.path.join(input_dir, "validation_set.json")


# In[5]:


# 0.3 Load the splited datasets from JSON files
with open(pilot_path, "r", encoding="utf-8") as f:
    pilot_data = json.load(f)
    
with open(development_path, "r", encoding="utf-8") as f:
    development_data = json.load(f)
    
with open(validation_path, "r", encoding="utf-8") as f:
    validation_data = json.load(f)


# In[7]:


# 0.4 Load the cases-per-study CSV file and standardize the study name
cases_df = pd.read_csv(CSV_path)
cases_df["Study Name"] = cases_df["0"].astype(str)


# In[9]:


# 1.1 Initialize 
replicate = Client(api_token=os.getenv("REPLICATE_API_TOKEN"))


# In[11]:


# 1.2 Set the prompt for code testing
prompt = """
You are a medical researcher, processing individual patient data from a publication database on N-Methyl-D-Aspartate Receptor Antibody Encephalitis (NMDARE) for an Individual Patient Data Meta-Analysis (IPDMA).

Please identify NMDARE patients in the input publications. 
The inclusion criteria for cases are: (1) patients with NMDARE with positive NMDAR antibodies in serum and/or cerebrospinal fluid (CSF); (2) their individual patient data on immunotherapy was provided.
The exclusion criteria for cases are: (1) patients with NMDARE preceded by central nervous system infection (e.g., herpes simplex virus encephalitis); (2) patients in large cohorts where individual patient details were not available.

Then extract the following features for each patient. Score clinical and treatment features as Yes if they occurred at any time during the first disease event (i.e. first clinical episode of NMDARE), disregarding any subsequent relapses. Extract continuous and categorical features values from the first event where applicable. However MonthsToFU, FinalMRS and Relapse regard the whole disease course.
- Female: sex (female or not)
- Age: age at disease onset (years)
- SpeechDys: presence of speech dysfunction (e.g. pressured speech, verbal reduction, mutism)
- Seizures: presence of seizures
- MvmtDis: presence of movement disorder (e.g. dyskinesias, or rigidity/abnormal postures)
- Obtunded: presence of decreased level of consciousness
- BrainstemDys: presence of autonomic dysfunction or central hypoventilation
- BehCogImp: presence of abnormal (psychiatric) behaviour or cognitive dysfunction
- ITU: whether admitted to an intensive care unit
- WorstMRSAcute: worst mRS (modified Rankin Scale (0-6) for assessing functional independence and disability in patients with neurological disorders) in the acute phase
- EEGSlow: presence of focal or diffuse slow or disorganised activity in EEG (Electroencephalography)
- EEGDeltaBrush: presence of delta brush in EEG
- EEGEpileptiform: presentation of epileptiform discharges in EEG
- CSFPleo: whether pleocytosis in CSF(Cerebrospinal Fluid) (i.e. Number of white blood cells per mm3 >= 5) 
- CSFWBCQuant: maximum number of white blood cells per mm3 in CSF 
- AbnMRIBrain: brain MRI (Magnetic Resonance Imaging) abnormal (increased T2/FLAIR parenchymal signal intensity or contrast enhancement)
- Tumour: presence of tumour
- DaysToIT: number of calendar days between date of first symptom onset of NMDARE and date of first immunotherapy was initiated. Follow the step-by-step reasoning process before giving the final result. Perform all reasoning internally before producing the final output and do not output any explanation in the final answer.
    (Reasoning Process:
    Step 1: Identify relevant time expressions in the text about: (1) the interval of onset to first immunotherapy; (2) date or time reference of symptom onset; (3) date or time reference of first immunotherapy initiation.
    Step 2: Classify the identified information into one of these cases: (A) interval is already given in days; (B) interval is given in weeks or months; (C) two specific dates are provided; (D) information is unavailable or incomplete.
    Step 3: Apply calculation rules: (for A) directly extract the number of days; (for B) convert to days (1 week = 7 days; 1 month = 30 days); (for C) calculate the difference in days between the two dates (calendar days, not rounded); (for D) assign null.
    Step 4: Output the result as an integer number of days without units, or null if missing.)
- ITWithin30Days: whether the time between onset and first immunotherapy ≤30 days?
- IT1stLineCombo: the specific combination of first line immunotherapies the patient received at their first disease event. (First-line immunotherapies include: corticosteroids [CS], Intravenous Immunoglobulin [IVIG], therapeutic plasma exchange [PE]/immunoadsorption [IA]; regardless of the order). <Categories include: "None", "CS only", "CS+IVIG", "CS+IVIG+PE/IA", "CS+PE/IA", "IVIG only", "IVIG+PE/IA", "PE/IA only">
- IT2ndLineRTX: initiation of Rituximab (RTX) at first disease event
- IT2ndLineCYC: initiation of Cyclophosphamide (CYC) at first disease event
- IT2ndLineBort: initiation of Bortezomib at first disease event
- IT2ndLineToc: initiation of Tocilizumab at first disease event
- ITMaintenanceMMF: initiation of Mycophenolate Mofetil (MMF) for any duration at first disease event
- ITMaintenanceAZA: initiation of Azathioprine (AZA) for any duration at first disease event
- ITMaintenanceMTX: initiation of Methotrexate (MTX) for any duration at first disease event
- IT6mSteroid: initiation of long-term steroids for ≥6 months at first disease event
- IT6mIVIG: initiation of long-term IVIG use for ≥6 months at first disease event
- MonthsToFU: number of months (as a float number) from the onset of NMDARE to the last follow-up. Follow the step-by-step reasoning process before giving the final result. Perform all reasoning internally before producing the final output and do not output any explanation in the final answer.
    (Reasoning Process:
    Step 1: Identify relevant time expressions in the text about: (1) the interval of onset to last follow-up; (2) date or time reference of symptom onset; (3) date or time reference of last follow-up.
    Step 2: Classify the identified information into one of these cases: (A) interval is already given in months; (B) interval is given in days or weeks or years; (C) two specific dates are provided; (D) information is unavailable or incomplete.
    Step 3: Apply calculation rules: (for A) directly extract the number of months; (for B) convert to months (30 days = 1 month; 1 week = 7 days → divide by 30 to convert to months; 1 year = 365 days → divide by 30 to convert to months); (for C) calculate the difference in calendar days between the two dates, then divide by 30 to get the number of months; (for D) assign null.
    Step 4: Output the result as a float number of months without units, or null if missing.)
- FinalMRS: mRS at last follow-up
- Relapse: presence of relapses of NMDARE

Output the result in JSON format. Each patient should be a dictionary with these fields. Binary features should be represented as 1 or 0, continuous features as numbers, and categorical features as strings. If the feature is unknown, unavailable or incomplete, leave it as null.
ONLY output the JSON list. Do NOT include any extra text, sentences, or explanations.
Output should be in the following format:
[
  {
    "Female": <Binary: Male encoded as 0, Female encoded as 1>,
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

Directly extract the "worst mRS in the acute phase" and "mRS at last follow-up" if they were provided in the publication text. Or if the value of mRS were not provided, evaluate the patient’s "worst mRS in the acute phase" and "mRS at last follow-up" using the modified Rankin Scale (mRS) scoring guides as follows:
Table 1 - Applicable to both adults and children:
    Score - Description - Comments
    0 - No symptoms
    1 - Non-disabling symptoms that do not interfere with the daily activity and playing/learning habits of the child - Playing/learning habits includes attending school or kindergarten
    2 - Minor symptoms that lead to some restriction in daily activity and playing/learning habits of the child, but do not interfere with the age-appropriate basic functions - Basic functions: drinking, eating, dressing, undressing, combing, washing, bathing; Symptoms: may include minor physical, cognitive and/or relational symptoms
    3 - Moderate symptoms that significantly interfere with the daily activity and playing/learning habits or prevent total independence in age-appropriate basic functions - Basic functions and symptoms as above
    4 - Moderately severe symptoms that clearly prevent independence in basic functions as would be appropriate for age, although patient does not need a constant attention - Basic functions and symptoms as above
    5 - Severely disabled, totally dependent, requires constant attention - Bed-bound; May have impaired consciousness, agitation, dysautonomia, severe movement disorder
    6 - Dead
Table 2 - For adults (=> 18 years old):
    0 - no symptoms
    1 - mild deficit without significant disability; capable of performing all usual duties and activities
    2 - mild disability; unable to carry out all previous activities, but able to manage own affairs without any assistance
    3 - moderate disability, the patient requires assistance with some activities; able to walk without another person’s help
    4 - unable to walk and attend to bodily needs without assistance
    5 - bedridden; incontinent, requiring constant nursing care
    6 - dead
Table 3 - For children (< 18 years old):
    0 - No symptoms at all
    1 - No significant disabilities despite symptoms in clinical examination; age appropriate behaviour and further development
    2 - Slight disability; unable to carry out all previous activities, but same independence as other age- and sex-matched children (no reduction of levels on the gross motor function scale)
    3 - Moderate disability; requiring some help, but able to walk without assistance; in younger patients adequate motor development despite mild functional impairment (reduction of one level on the gross motor function scale)
    4 - Moderately severe disability; unable to walk without assistance; in younger patients reduction of at least 2 levels on the gross motor function scale
      - GMFCS (before 2 years old): Level III = unable to sit without trunk support, unable to crawl or pull to stand
      - GMFCS (>= 2 years old): Level III = requires a hand-held mobility device or adult support to walk
    5 - severe disability; bedridden, requiring constant nursing care and attention
    6 - dead

Ensure that all generated values are internally consistent with the following rules (If a rule is violated, adjust the values accordingly): (1) if WorstMRSAcute = 6, FinalMRS must be 6 or null; (2) if DaysToIT < 30, ITWithin30Days must be 1; (3) if EEGDeltaBrush = 1, EEGSlow must be 1; (4) if CSFWBCQuant >= 5, CSFPleo must be 1; (5) if DaysToIT is assigned a number and/or ItWithin30Days = 1, IT1stLineCombo cannot be "None"; (6) if DaysToIT is missing, ITWithin30Days must be 0
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
    start_time = time.time()

    response = replicate.run(
        "meta/llama-4-scout-instruct",
        input={
            "prompt": prompt + " The input publication for data extraction is: " + text,
            "temperature": 0,
            "max_output_tokens": 4000
        }
    )

    end_time = time.time()
    duration = end_time - start_time
    
    output_text = "".join(response)
    
    match = re.search(r"\[(.*?)\]", output_text, re.DOTALL)
    if match:
        output_json = match.group(1).strip()
    else:
        match = re.search(r"\{(.*?)\}", output_text, re.DOTALL)
        output_json = "{" + match.group(1).strip() + "}"
    
    data = json.loads("[" + output_json + "]")                                                                 ### Parse the JSON string to Python data structures                   
    IPD = data[0]
        
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

    response = replicate.run(
        "meta/llama-4-scout-instruct",
        input={
            "prompt": prompt + " The input publication for data extraction is: " + text,
            "temperature": 0,
            "max_output_tokens": 4000
        }
    )

    end_time = time.time()
    duration = end_time - start_time

    output_text = "".join(response)
    
    all_matches = re.findall(r"\{(.*?)\}", output_text, re.DOTALL)
    data = [json.loads("{" + match.strip()+ "}") for match in all_matches]
    
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

