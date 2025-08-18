#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 0.1 Import required libraries
import pandas as pd
import numpy as np
import os
import re
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    mean_squared_error, log_loss
)


# In[3]:


# 0.2 Set path variables
input_dir = 'LLM_extracted_IPD'
output_dir = 'accuracy_metrics'


# In[5]:


# 0.3 Load the LLM output and human-annotated data
pilot_path = os.path.join(input_dir, "pilot_output.csv")
df_pilot = pd.read_csv(pilot_path)
development_path = os.path.join(input_dir, "development_output.csv")
df_development = pd.read_csv(development_path)

CSV_path = 'Cases per paper.csv'
cases_df = pd.read_csv(CSV_path)
cases_df["Study Name"] = cases_df["0"].astype(str)

target_path_xls = 'NMDARE SR for STATISTICS 2021.01.26 1551 pts (excluding 16 Ab-neg and 83 post-infect) FINAL BEFORE CLEANING.xls'
df_human_xls = pd.read_excel(target_path_xls)
target_path_csv = 'MasterNew2021v2 FINAL BEFORE IMPUTATION.csv'
df_human_csv = pd.read_csv(target_path_csv)


# In[7]:


# 1.1 Merge human-annotated datasets to combine metadata and main data
df_xls_id = df_human_xls[["ID","Time from onset of anti-NMDARE to last follow-up (months)", "Relapses of anti-NMDAR encephalitis?\n\n0=No\n1=Yes\nblank=Not available", "First author - Last author, year"]]
df_human = df_human_csv.merge(df_xls_id, left_on="ID", right_on="ID", how='left')
df_human = df_human.rename(columns={
    "Time from onset of anti-NMDARE to last follow-up (months)": "MonthsToFU",
    "Relapses of anti-NMDAR encephalitis?\n\n0=No\n1=Yes\nblank=Not available": "Relapse"
})
df_human.head()


# In[9]:


# 1.2 Categorize the features by data type and predefine the classes of the categorical feature
binary_features = ["Female", "SpeechDys", "Seizures", "MvmtDis", "Obtunded", "BrainstemDys", "BehCogImp", "ITU", "EEGSlow", "EEGDeltaBrush",
                  "EEGEpileptiform", "CSFPleo", "AbnMRIBrain", "Tumour", "ITWithin30Days", "IT2ndLineRTX", "IT2ndLineCYC", "IT2ndLineBort", "IT2ndLineToc", "ITMaintenanceMMF",
                  "ITMaintenanceAZA", "ITMaintenanceMTX", "IT6mSteroid", "IT6mIVIG", "Relapse"]
continuous_features = ["Age", "WorstMRSAcute", "CSFWBCQuant", "DaysToIT", "MonthsToFU", "FinalMRS"]
categorical_features = ["IT1stLineCombo"]

fixed_classes = [
"missingness", "None", "CS only", "CS+IVIG", "CS+IVIG+PE/IA",
"CS+PE/IA", "IVIG only", "IVIG+PE/IA", "PE/IA only"
]


# In[11]:


# 1.3 Calculate the 95th percentiles for specific columns and construct a penalty mapping dictionary
to_cal_95cent = ["CSFWBCQuant", "DaysToIT", "MonthsToFU"]
percentiles_95 = df_human[to_cal_95cent].quantile(0.95)
print(percentiles_95)

penalty_map = {
    "Age": 16,
    "WorstMRSAcute": 6,
    "CSFWBCQuant": percentiles_95["CSFWBCQuant"],
    "DaysToIT": percentiles_95["DaysToIT"],
    "MonthsToFU": percentiles_95["MonthsToFU"],
    "FinalMRS": 6
}


# In[13]:


# 2.1 Define a function to match IPD by features between LLM-generated data and human-annotated data using a one-to-one matching rule for single-case series and the Hungarian algorithm for case series.

# 2.1.1 Calculate the standard deviation of each categorical feature in the human-annotated dataset to later standardize distances
cont_std_dict = df_human[continuous_features].std().to_dict()

# 2.1.2 Define a function to compute the distance between two cases
def compute_case_distance(row_llm, row_human, binary_features, continuous_features, categorical_features):
    """
    Calculate a distance metric between one LLM-extracted case and one human-annotated case.    
    Args:
        - row_llm (pandas Series): one case from LLM-extracted dataset
        - row_human (pandas Series): one case from human-annotated dataset
        - binary_features: list of features considered binary
        - continuous_features: list of features considered continuous
        - categorical_features: list of features considered categorical
    Returns:
        - dist (float): the total computed distance between the two cases
    """
    dist = 0.0
    
    for f in binary_features:                                                 ### Compute binary features: add 1 if values differ
        val1, val2 = row_llm.get(f), row_human.get(f)
        if pd.notna(val1) and pd.notna(val2) and val1 != val2:
            dist += 1

    for f in continuous_features:                                            ### Compute continuous features: add the absolute difference
        val1, val2 = row_llm.get(f), row_human.get(f)
        if pd.notna(val1) and pd.notna(val2):
            std_f = cont_std_dict.get(f, 1)
            dist += abs(val1 - val2) / std_f

    for f in categorical_features:                                          ### Compute categorical features: add 1 if categories differ
        val1, val2 = row_llm.get(f), row_human.get(f)
        if pd.notna(val1) and pd.notna(val2) and val1 != val2:
            dist += 1
    
    return dist

# 2.1.3 Define a function to match LLM-extracted cases to human-annotated cases by minimizing total feature distance
def create_matches(df_llm, df_human):
    """
    Match LLM cases with corresponding human cases on a per-study basis.
    Args:
        - df_llm (pd.DataFrame): containing LLM-extracted IPD
        - df_human (pd.DataFrame): containing human-annotated IPD   
    Returns:
        matches (list of dict): Each dict contains:
            - 'study': study identifier
            - 'patient_idx': index of the patient record in df_llm
            - 'y_true': dict of human-annotated feature values
            - 'y_pred': dict of LLM-predicted feature values
    """
    matches = []
    df_llm = df_llm.reset_index()

    for study in df_llm["Study"].unique():                                                           ### Extract subsets of cases from both LLM and human datasets for the current study and index the subsets
        sub_llm = df_llm[df_llm["Study"] == study].reset_index()
        sub_human = df_human[df_human["First author - Last author, year"] == study].reset_index()

        m, n = len(sub_llm), len(sub_human)

        if m == 1 and n == 1:                                                                        ### For single-case reports, match directly
            row_llm = sub_llm.iloc[0]
            row_human = sub_human.iloc[0]

        else:
            cost_matrix = np.zeros((m, n))                                                           ### Build cost matrix of pairwise distances
            for i, (_, row_llm) in enumerate(sub_llm.iterrows()):
                for j, (_, row_human) in enumerate(sub_human.iterrows()):
                    cost_matrix[i, j] = compute_case_distance(
                        row_llm, row_human, 
                        binary_features, continuous_features, categorical_features
                    )
    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)                                    ### Solve assignment problem to find the optimal one-to-one matching

            for i, j in zip(row_ind, col_ind):
                row_llm = sub_llm.iloc[i]
                row_human = sub_human.iloc[j]
            
            unmatched_llm = set(range(m)) - set(row_ind)                                             ### Identify unmatched cases
            unmatched_human = set(range(n)) - set(col_ind)
            if unmatched_llm:
                print(f" ⚠️ Unmatched LLM-extracted case indices for study {study}: {list(unmatched_llm)}")
            if unmatched_human:
                print(f" ⚠️ Unmatched human-annotated case indices for study {study}: {list(unmatched_human)}")          

        y_true, y_pred = {}, {}
        for feature in binary_features+continuous_features+categorical_features:          ### For each feature, map matched feature values from LLM and human dataset for evaluation
            y_pred[feature] = row_llm.get(feature)
            y_true[feature] = row_human.get(feature)
            
        matches.append({
            "study": study,
            "patient_idx": row_human.get("ID"),
            "y_true": y_true,
            "y_pred": y_pred
        })
        
    return matches


# In[15]:


# 2.2 Define a function to evaluate the agreement on missing for a specific feature
def evaluate_missing_value_agreement(matches, feature):
    """
    Evaluate missingness as a binary classification problem for a specific feature.
    Args:
        - matches (list of dict): A list of matched records, each containing:
                                - "y_true": dict of human-annotated feature values
                                - "y_pred": dict of model-generated feature values
        - feature (str): The name of the feature to evaluate.
    Returns:
        - precision (float or None): TP / (TP + FP), predicted missing correct rate
        - recall (float or None): TP / (TP + FN), detected missing rate
    """
    missing_total = 0
    TP = 0
    FP = 0
    FN = 0

    for m in matches:
        tv = m["y_true"].get(feature)
        pv = m["y_pred"].get(feature)

        tv_missing = tv is None or pd.isna(tv) or (isinstance(tv, str) and tv.strip() == "")         ### Define missingness conditions
        pv_missing = pv is None or pd.isna(pv) or (isinstance(pv, str) and pv.strip() == "")

        if tv_missing or pv_missing:
            missing_total += 1

        if pv_missing and tv_missing:
            TP += 1
        elif pv_missing and not tv_missing:
            FP += 1
        elif not pv_missing and tv_missing:
            FN += 1

        missing_precision = TP / (TP + FP) if (TP + FP) > 0 else None
        missing_recall = TP / (TP + FN) if (TP + FN) > 0 else None
        
    if missing_total > 0:
        print(f" Missingness evaluation for '{feature}': Precision: {missing_precision}; Recall: {missing_recall}")
    else:
        agreement_rate = None                                                                         ### None if no missing cases detected
        print(f" Missingness evaluation for '{feature}': No missing cases detected.")

    return missing_precision, missing_recall, TP, FP, FN


# In[17]:


# 2.3 Define a function for accuracy evaluation
def evaluate_matrics(matches):
    """
    Evaluate the accuracy of generated patient data by comparing it with human-annotated data.
    Evaluation is performed separately for binary, continuous, and categorical features using appropriate metrics.
    Args:
        - matches (list of dict): A list of matched records from model output and annotations.
                                  Each entry contains:
                                    - "study": source study
                                    - "patient_idx": patient index
                                    - "y_true": human-annotated patient data (dict)
                                    - "y_pred": model-generated patient data (dict)

    Prints:
        - Precision, Recall, F1 Score for binary features
        - Mean Squared Error (MSE) for continuous features
        - Cross-Entropy (Log Loss) for categorical features
        - Precision, Recall for missingness
    """    
    # 2.3.1 Create lists to contain accuracy metrics
    binary_results = []
    continuous_results = []
    categorical_results = []
    missingness_TP = missingness_FP = missingness_FN = 0

    # 2.3.2 Accuracy evaluation for binary features
    for feature in binary_features:
        y_true, y_pred = [], []

        for m in matches:
            tv = m["y_true"].get(feature)
            pv = m["y_pred"].get(feature)

            if tv is not None and not pd.isna(tv):
                if pv is None or pd.isna(pv):                    ### If the predicted value (pv) is missing or NaN, assign it as wrong extraction
                    pv = 1 - tv 
                y_true.append(tv)
                y_pred.append(pv)

        if not y_true:                                          ### Skip feature if no valid annotations are available
            print(f"\n ⚠️ Skipping {feature} - lack of valid annotation.")
            continue

        n = len(y_true)                                         ### Count valid cases where the true value exists

        pos_true = sum(y_true)
        pos_pred = sum(y_pred)
        precision = recall = f1 = None                          ### Metrics are set as undefined if no instances of the positive class
        
        if pos_pred > 0:
            precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        if pos_true > 0:
            recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        if pos_true > 0 and pos_pred > 0:
            f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        
        bal_acc = balanced_accuracy_score(y_true, y_pred)       ### Compute Balanced Accuracy to account for sample imbalance in the analysis

        print(f"\n=== Binary Feature: {feature} ===")
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Balanced Accuracy:", bal_acc)
        
        missing_precision, missing_recall, TP, FP, FN = evaluate_missing_value_agreement(matches, feature)        ### Check agreement of missing for the processing feature
        binary_results.append({
            "Feature": feature,
            "Type": "Binary",
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Balanced Accuracy": bal_acc,
            "Missing Precision": missing_precision, 
            "Missing Recall": missing_recall,
            "Number of Valid Cases": n
        })

        missingness_TP = missingness_TP + TP
        missingness_FP = missingness_FP + FP
        missingness_FN = missingness_FN + FN

    # 2.3.3 Accuracy evaluation for continuous features
    for feature in continuous_features:
        y_true, y_pred = [], []

        for m in matches:
            tv = m["y_true"].get(feature)
            pv = m["y_pred"].get(feature)

            if tv is not None and not pd.isna(tv):                                ### If the predicted value (pv) is missing or NaN, fill it using the true value plus a penalty
                if pv is None or pd.isna(pv):
                    pv = tv + penalty_map.get(feature)
                y_true.append(tv)
                y_pred.append(pv)

        if not y_true:                                          ### Skip feature if no valid annotations are available
            print(f"\n ⚠️ Skipping {feature} - lack of valid annotation.")
            continue

        n = len(y_true)
            
        mse = mean_squared_error(y_true, y_pred)
        
        var_y = np.var(y_true, ddof=1)
        nmse = mse / var_y if var_y > 0 else np.nan                 # Compute the normalized mean squared error (NMSE) to better account for the varience of the true values 

        print(f"\n=== Continuous Feature: {feature} ===")
        print("MSE:", mse)
        print("NMSE:", nmse)
        
        missing_precision, missing_recall, TP, FP, FN = evaluate_missing_value_agreement(matches, feature)
        continuous_results.append({
            "Feature": feature,
            "Type": "Continuous",
            "MSE": mse,
            "NMSE": nmse,
            "Missing Precision": missing_precision, 
            "Missing Recall": missing_recall,
            "Number of Valid Cases": n
        })
        
        missingness_TP = missingness_TP + TP
        missingness_FP = missingness_FP + FP
        missingness_FN = missingness_FN + FN

    # 2.3.4 Accuracy evaluation for text-based features
    for feature in categorical_features:
        y_true_raw, y_pred_raw = [], []

        for m in matches:
            tv = m["y_true"].get(feature)
            pv = m["y_pred"].get(feature)

            if tv is not None and not pd.isna(tv):
                if pv is None or pd.isna(pv):                     ### If the predicted value (pv) is missing or NaN, assign it as wrong extraction
                    pv = "missingness"
                y_true_raw.append(str(tv).lower())
                y_pred_raw.append(str(pv).lower())

        if not y_true_raw:                                     
            print(f"\n ⚠️ Skipping {feature} - lack of valid annotation.")
            continue                                            ### Skip feature if no valid annotations are available

        if len(set(y_true_raw)) < 2 or len(set(y_pred_raw)) < 2 :   
            print(f"\n ⚠️ Skipping log loss for {feature} — only one class found.")
            continue                                            ### Log loss requires at least 2 classes

        le = LabelEncoder()
        le.fit([c.lower() for c in fixed_classes])                                     ### Encode text labels into numeric classes
        y_true = le.transform(y_true_raw)
        y_pred = le.transform(y_pred_raw)
        
        num_classes = len(le.classes_)
        y_pred_prob = np.eye(num_classes)[y_pred]               ### One-hot encoding of predictions

        ce_loss = log_loss(y_true, y_pred_prob, labels=range(num_classes))

        print(f"\n=== Text Feature: {feature} ===")
        print("Cross-Entropy (Log Loss):", ce_loss)
        
        missing_precision, missing_recall, TP, FP, FN = evaluate_missing_value_agreement(matches, feature)
        categorical_results.append({
            "Feature": feature,
            "Type": "Categorical",
            "LogLoss": ce_loss,
            "Missing Precision": missing_precision, 
            "Missing Recall": missing_recall
        })
    
        missingness_TP = missingness_TP + TP
        missingness_FP = missingness_FP + FP
        missingness_FN = missingness_FN + FN

    total_missing_precision = missingness_TP / (missingness_TP + missingness_FP) if (missingness_TP + missingness_FP) > 0 else None
    total_missing_recall = missingness_TP / (missingness_TP + missingness_FN) if (missingness_TP + missingness_FN) > 0 else None

    return binary_results, continuous_results, categorical_results, total_missing_precision, total_missing_recall


# In[19]:


# 2.4 Define functions for global metrics

# 2.4.1 Define a function to compute the weighted average of F1 and Balanced Accuracy
def weighted_average_binary(metrics):
    """
    Compute the weighted average of evaluation metrics for binary features.    
    Args:
        metrics (list of dict): Each dict should contain keys:
            - "Number of Valid Cases" (int): the number of valid cases for this feature
            - "F1" (float): F1 score
            - "Balanced Accuracy" (float): balanced accuracy score   
    Returns:
        tuple: (weighted_f1, weighted_bal_acc)
            - weighted_f1 (float): weighted average of F1 scores
            - weighted_bal_acc (float): weighted average of balanced accuracy scores
    """
    weights = np.array([m["Number of Valid Cases"] for m in metrics])
    f1_vals = np.array([0 if m["F1"] is None else m["F1"] for m in metrics])                  ###Replace None with 0
    bal_acc_vals = np.array([m["Balanced Accuracy"] for m in metrics])
    
    weighted_f1 = np.sum(f1_vals * weights) / np.sum(weights)
    weighted_bal_acc = np.sum(bal_acc_vals * weights) / np.sum(weights)
    
    return weighted_f1, weighted_bal_acc

# 2.4.2 Define a function to compute the weighted average of Normalized MSE
def weighted_average_continuous(metrics):
    """
    Compute the weighted average of evaluation metrics for continuous features.
    Args:
        metrics (list of dict): Each dict should contain keys:
            - "Number of Valid Cases" (int): the number of valid cases for this feature
            - "NMSE" (float): normalized mean squared error
    Returns:
        weighted_nmse (float): weighted average of NMSE values
    """
    weights = np.array([m["Number of Valid Cases"] for m in metrics])
    nmse_vals = np.array([m["NMSE"] for m in metrics])
    
    weighted_nmse = np.sum(nmse_vals * weights) / np.sum(weights)
    
    return weighted_nmse


# In[21]:


# 3.1 Conduct accuracy evaluation on pilot set and save the results
binary_metrics_pilot, continuous_metrics_pilot, categorical_metrics_pilot, missing_precision_pilot, missing_recall_pilot = evaluate_matrics(create_matches(df_pilot, df_human))

pd.DataFrame(binary_metrics_pilot).to_csv(os.path.join(output_dir, "binary_metrics_pilot.csv"), index=False)
pd.DataFrame(continuous_metrics_pilot).to_csv(os.path.join(output_dir, "continuous_metrics_pilot.csv"), index=False)
pd.DataFrame(categorical_metrics_pilot).to_csv(os.path.join(output_dir, "categorical_metrics_pilot.csv"), index=False)


# In[23]:


# 3.2 Compute and display weighted average metrics for the pilot set
weighted_f1_pilot, weighted_bal_acc_pilot = weighted_average_binary(binary_metrics_pilot)
weighted_nmse_pilot = weighted_average_continuous(continuous_metrics_pilot)

print('\n=== Weighted average metrics of binary features on the pilot set:')
print(f'Weighted F1 Score: {weighted_f1_pilot:.3f}\nWeighted Balanced Accuracy: {weighted_bal_acc_pilot:.3f}')

print('\n=== Weighted average metrics of continuous features on the pilot set:')
print(f'Weighted NMSE: {weighted_nmse_pilot:.3f}')

print('\n=== Micro average metrics of missingness on the pilot set:')
print(f'Micro averaged precision of missingness: {missing_precision_pilot:.3f}\nMicro averaged recall of missingness: {missing_recall_pilot:.3f}')


# In[25]:


# 4.1 Conduct accuracy evaluation on development set and save the results
binary_metrics_development, continuous_metrics_development, categorical_metrics_development, missing_precision_development, missing_recall_development = evaluate_matrics(create_matches(df_development, df_human))

pd.DataFrame(binary_metrics_development).to_csv(os.path.join(output_dir, "binary_metrics_development.csv"), index=False)
pd.DataFrame(continuous_metrics_development).to_csv(os.path.join(output_dir, "continuous_metrics_development.csv"), index=False)
pd.DataFrame(categorical_metrics_development).to_csv(os.path.join(output_dir, "categorical_metrics_development.csv"), index=False)


# In[27]:


# 4.2 Compute and display weighted average metrics for the development set
weighted_f1_development, weighted_bal_acc_development = weighted_average_binary(binary_metrics_development)
weighted_mse_development = weighted_average_continuous(continuous_metrics_development)

print('\n=== Weighted average metrics of binary features on the development set:')
print(f'Weighted F1 Score: {weighted_f1_development:.3f}\nWeighted Balanced Accuracy: {weighted_bal_acc_development:.3f}')

print('\n=== Weighted average metrics of continuous features on the development set:')
print(f'Weighted NMSE: {weighted_mse_development:.3f}')

print('\n=== Micro average metrics of missingness on the development set:')
print(f'Micro averaged precision of missingness: {missing_precision_development:.3f}\nMicro averaged recall of missingness: {missing_recall_development:.3f}')

