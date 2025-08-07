#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 0.1 Import required libraries
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    mean_squared_error, log_loss
)


# In[4]:


# 0.2 Set path variables
input_dir = 'LLM_extracted_IPD'
output_dir = 'accuracy_metrics'


# In[6]:


# 0.3 Load the LLM output and human annotation data
pilot_path = os.path.join(input_dir, "pilot_output.csv")
df_llm = pd.read_csv(pilot_path)

xls_path = 'NMDARE SR for STATISTICS 2021.01.26 1551 pts (excluding 16 Ab-neg and 83 post-infect) FINAL BEFORE CLEANING.xls'
df_human = pd.read_excel(xls_path)


# In[8]:


# 0.4 Define a field mapping dictionary (llm : human)
field_mapping = {
    "Age (years)": "Age at disease onset (years)\n\nblank=Not available",
    "Sex": "Gender\n\n0=Male\n1=Female\nblank=Not available",
    "Tumor": "Tumor\n\n0=No\n1=Yes\nblank=Not available",
    "Number of Main Group of Symptoms": "Number of main group of symptoms",
    "Length of Hospital Stay (days)": "Lenght of hospital stay (days) \n\nblank=Not available",
    "T2 and FLAIR Hyperintensities": "T2 and FLAIR hyperintensities"
}

# 0.5 Categorize the features by data type
binary_features = ["Sex", "Tumor"]
continuous_features = ["Age (years)", "Number of Main Group of Symptoms", "Length of Hospital Stay (days)"]
categorical_features = ["T2 and FLAIR Hyperintensities"]


# In[10]:


# 1.1 Define a function to match IPD by features between LLM-generated data and human-annotated data
def create_matches(df_llm, df_human):
    """
    Create a list of matched patient records
    Args:
        df_llm (pd.DataFrame): DataFrame containing LLM-generated data.
        df_human (pd.DataFrame): DataFrame containing human-annotated data.
    Returns:
        matches (list of dict): Each dict contains:
            - 'study': study identifier
            - 'patient_idx': index of the patient record in df_llm
            - 'y_true': dict of human-annotated feature values
            - 'y_pred': dict of LLM-predicted feature values
    """
    matches = []

    for idx, row_llm in df_llm.iterrows():
        y_true, y_pred = {}, {}
        
        study = row_llm["Study"]
        row_human = df_human[df_human["First author - Last author, year"] == study].iloc[0]

        for llm_field, human_field in field_mapping.items():                         ### For each feature, map values from LLM and human data
            val_llm = row_llm.get(llm_field)
            val_human = row_human.get(human_field)

            y_pred[llm_field] = val_llm
            y_true[llm_field] = val_human
            
        matches.append({
            "study": study,
            "patient_idx": idx,
            "y_true": y_true,
            "y_pred": y_pred
        })
        
    return matches


# In[12]:


# 1.2 Define a function to evaluate the agreement on missing for a specific feature
def evaluate_missing_value_agreement(matches, feature):
    """
    Evaluate the agreement between predicted and true values on missing data.
    Args:
        matches (list of dict): A list of matched records, each containing:
                                - "y_true": dict of human-annotated feature values
                                - "y_pred": dict of model-generated feature values
        feature (str): The name of the feature to evaluate missing value agreement on.
    Returns:
        missing_total (int): Total number of record-pairs where either true or predicted value is missing.
        missing_match (int): Number of record-pairs where both true and predicted values are missing.
        agreement_rate (float or None): Proportion of matching missing among all missing cases.
    """
    missing_total, missing_match = 0, 0

    for m in matches:
        tv = m["y_true"].get(feature)
        pv = m["y_pred"].get(feature)

        tv_missing = tv is None or pd.isna(tv) or (isinstance(tv, str) and tv.strip() == "")          ### Define missingness conditions
        pv_missing = pv is None or pd.isna(pv) or (isinstance(pv, str) and pv.strip() == "")

        if tv_missing or pv_missing:
            missing_total += 1
            if tv_missing and pv_missing:
                missing_match += 1

    if missing_total > 0:
        agreement_rate = missing_match / missing_total
        print(f" Missing Value Agreement for '{feature}': {missing_match} / {missing_total} matched ({agreement_rate:.1%})")
    else:
        agreement_rate = None                                                                         ### None if no missing cases detected
        print(f" Missing Value Agreement for '{feature}': No missing cases detected.")

    return missing_total, missing_match, agreement_rate


# In[14]:


# 1.3 Define a function for accuracy evaluation
def evaluate_matrics(matches):
    """
    Evaluate the accuracy of generated patient data by comparing it with human-annotated data.
    Evaluation is performed separately for binary, continuous, and categorical features using appropriate metrics.
    Args:
        matches (list of dict): A list of matched records from model output and annotations.
                                Each entry contains:
                                    - "study": source study
                                    - "patient_idx": patient index
                                    - "y_true": human-annotated patient data (dict)
                                    - "y_pred": model-generated patient data (dict)

    Prints:
        - Precision, Recall, F1 Score for binary features
        - Mean Squared Error (MSE) for continuous features
        - Cross-Entropy (Log Loss) for categorical features
    """    
    # 1.3.1 Create lists to contain accuracy metrics
    binary_results = []
    continuous_results = []
    categorical_results = []

    # 1.3.2 Accuracy evaluation for binary features
    for feature in binary_features:
        y_true, y_pred = [], []

        for m in matches:
            tv = m["y_true"].get(feature)
            pv = m["y_pred"].get(feature)

            if tv is not None and pv is not None:               ### Only compare when both values are valid
                y_true.append(tv)
                y_pred.append(pv)

        if not y_true:                                          ### Skip feature if no valid annotations are available
            print(f"\n ⚠️ Skipping {feature} - lack of valid annotation.")
            continue

        precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

        print(f"\n=== Binary Feature: {feature} ===")
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        
        missing_total, missing_match, agreement_rate = evaluate_missing_value_agreement(matches, feature)        ### Check agreement of missing for the processing feature
        binary_results.append({
            "Feature": feature,
            "Type": "Binary",
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Missing Total": missing_total,
            "Missing Match": missing_match,
            "Missing Match %": agreement_rate
        })

    # 1.3.3 Accuracy evaluation for continuous features
    for feature in continuous_features:
        y_true, y_pred = [], []

        for m in matches:
            tv = m["y_true"].get(feature)
            pv = m["y_pred"].get(feature)

            if tv is not None and pv is not None and not pd.isna(tv) and not pd.isna(pv):  
                y_true.append(tv)
                y_pred.append(pv)                               ### Only compare when both values are valid

        if not y_true:                                          ### Skip feature if no valid annotations are available
            print(f"\n ⚠️ Skipping {feature} - lack of valid annotation.")
            continue

        mse = mean_squared_error(y_true, y_pred)
        print(f"\n=== Continuous Feature: {feature} ===")
        print("MSE:", mse)
        
        missing_total, missing_match, agreement_rate = evaluate_missing_value_agreement(matches, feature)
        continuous_results.append({
            "Feature": feature,
            "Type": "Continuous",
            "MSE": mse,
            "Missing Total": missing_total,
            "Missing Match": missing_match,
            "Missing Match %": agreement_rate
        })

    # 1.3.4 Accuracy evaluation for text-based features
    for feature in categorical_features:
        y_true_raw, y_pred_raw = [], []

        for m in matches:
            tv = m["y_true"].get(feature)
            pv = m["y_pred"].get(feature)

            if tv and pv:                                       ### Filter out missing values and normalize text to lowercase
                y_true_raw.append(str(tv).lower())
                y_pred_raw.append(str(pv).lower())

        if not y_true_raw:                                     
            print(f"\n ⚠️ Skipping {feature} - lack of valid annotation.")
            continue                                            ### Skip feature if no valid annotations are available

        if len(set(y_true_raw)) < 2 or len(set(y_pred_raw)) < 2 :   
            print(f"\n ⚠️ Skipping log loss for {feature} — only one class found.")
            continue                                            ### Log loss requires at least 2 classes

        all_classes = list(set(y_true_raw + y_pred_raw))
        le = LabelEncoder()
        le.fit(all_classes)                                     ### Encode text labels into numeric classes
        y_true = le.transform(y_true_raw)
        y_pred = le.transform(y_pred_raw)
        
        num_classes = len(le.classes_)
        y_pred_prob = np.eye(num_classes)[y_pred]               ### One-hot encoding of predictions

        ce_loss = log_loss(y_true, y_pred_prob, labels=range(num_classes))

        print(f"\n=== Text Feature: {feature} ===")
        print("Classes:", le.classes_)
        print("Cross-Entropy (Log Loss):", ce_loss)
        
        missing_total, missing_match, agreement_rate = evaluate_missing_value_agreement(matches, feature)
        categorical_results.append({
            "Feature": feature,
            "Type": "Categorical",
            "LogLoss": ce_loss,
            "Classes": ", ".join(le.classes_),
            "Missing Total": missing_total,
            "Missing Match": missing_match,
            "Missing Match %": agreement_rate
        })
        
    return binary_results, continuous_results, categorical_results


# In[16]:


# 2.1 Conduct accuracy evaluation on pilot set and save the results
binary_metrics, continuous_metrics, categorical_metrics = evaluate_matrics(create_matches(df_llm, df_human))

pd.DataFrame(binary_metrics).to_csv(os.path.join(output_dir, "binary_metrics.csv"), index=False)
pd.DataFrame(continuous_metrics).to_csv(os.path.join(output_dir, "continuous_metrics.csv"), index=False)
pd.DataFrame(categorical_metrics).to_csv(os.path.join(output_dir, "categorical_metrics.csv"), index=False)

