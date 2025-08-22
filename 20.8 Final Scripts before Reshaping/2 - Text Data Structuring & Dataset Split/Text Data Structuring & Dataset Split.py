#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 0.1 Install dependencies for handling PDF, DOCX, and Excel files
get_ipython().system('pip install pdfplumber python-docx openpyxl xlrd')


# In[3]:


# 0.2 Import required libraries
import pandas as pd
import os
import re
import pdfplumber
import docx
import openpyxl
import xlrd
import random
from pathlib import Path
from collections import defaultdict


# In[5]:


# 0.3 Set path variables
CSV_path = 'Cases per paper.csv'
input_dir = 'manual_edited'
output_dir = 'preprocess_outputs'

matched_path = os.path.join(input_dir, "matched_final.xlsx")
group_path = os.path.join(input_dir, "study_groups_review_final.xlsx")


# In[7]:


# 1.1 Load matched study-group pairs and group-file metadata
matched_df = pd.read_excel(matched_path)
group_df = pd.read_excel(group_path)


# In[9]:


# 1.2 Map each study name to its corresponding list of file entries based on matched group
study_to_file = defaultdict(list)

for _, row in matched_df.iterrows():
    study_name = row["Study Name"]
    matched_group = row["Matched Group Name"]
    matched_entries = group_df[group_df["Study Group Key"] == matched_group]
    study_to_file[study_name] = matched_entries.to_dict(orient="records")


# In[11]:


# 1.3 Summarize the total number of studies and associated files
total_studies = len(study_to_file)
total_files = sum(len(v) for v in study_to_file.values())

print(f"Total Number of Studies：{total_studies}")
print(f"Total Number of Files to be Read：{total_files}")


# In[13]:


# 1.4 Identify studies that are linked to zero or multiple files (for manual inspection)
for study, entries in study_to_file.items():
    if len(entries) != 1:
        print(f"{study}: {len(entries)} files")

### After manual inspection, it was found that all files were correctly associated with their corresponding studies.


# In[15]:


# 2.1 Define the functions to extract text content from different document formats

# 2.1.1 Function to extract text data from a PDF file
def extract_text_from_pdf(filepath):
    """
    Extract all text content from a PDF file.
    Parameters:
        filepath (str): The path to the PDF file.
    Returns:
        str: Extracted text from all pages.
    """
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f" ⚠️ Failed to read PDF: {filepath}, Error: {e}")          ### Print an error message if reading fails
    return text

# 2.1.2 Function to extract text data from a DOCX (Word) file
def extract_text_from_docx(filepath):
    """
    Extract all text content from a DOCX (Word) file.
    Parameters:
        filepath (str): The path to the DOCX file.
    Returns:
        str: Extracted text concatenated from all paragraphs.
    """
    text = ""
    try:
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f" ⚠️ Failed to read DOCX: {filepath}, Error: {e}")        ### Print an error message if reading fails
    return text

# 2.1.3 Function to extract text data from a XLS (Excel) file
def extract_text_from_xls(filepath):
    """
    Extract all text content from an XLS (Excel) file.
    Parameters:
        filepath (str): The path to the XLS file.
    Returns:
        str: Extracted text from all sheets and rows, cells joined by tabs and rows by newlines.
    """
    text = ""
    try:
        book = xlrd.open_workbook(filepath)
        for sheet in book.sheets():
            for row_idx in range(sheet.nrows):
                row = sheet.row_values(row_idx)
                row_text = [str(cell) for cell in row if cell]
                if row_text:
                    text += "\t".join(row_text) + "\n"
    except Exception as e:
        print(f" ⚠️ Failed to read XLS: {filepath}, Error: {e}")         ### Print an error message if reading fails
    return text


# In[17]:


# 2.2 Extract all text in every file associated with each study
study_to_text = {}

for i, (study_name, entry_list) in enumerate(study_to_file.items(), 1):
    all_text = ""
    print(f"\nProcessing Study #{i}: {study_name}")

    if not entry_list:                                                             ### If no associated files, report a warning and assign empty string
        print(f" ⚠️ No study group found for study: {study_name}")
        study_to_text[study_name] = ""
        continue
        
    for entry in entry_list:
        path_data = entry["Full Path"]
        file_name = entry["Filename"]
        file_path = os.path.join(input_dir, path_data)

        if not os.path.exists(file_path):                                          ### If the file doesn't exist or path is invalid, print a warning and skip
            print(f" ⚠️ Invalid file path: {file_path}")
            continue
            
        print(f" Reading: {file_name}")

        ext = Path(file_path).suffix.lower()                                       ### Get file extension in lowercase
        if ext == ".pdf":                                                          ### Extract text based on file type
            all_text += extract_text_from_pdf(file_path) + "\n"
        elif ext == ".docx":
            all_text += extract_text_from_docx(file_path) + "\n"
        elif ext == ".xls":
            all_text += extract_text_from_xls(file_path) + "\n"
        else:
            print(f" ⚠️ Skipped unsupported file type: {file_name}")               ### If file type is unsupported, print a warning and skip

    study_to_text[study_name] = all_text.strip()                                   ### Store the combined text under the study name


# In[19]:


# 2.3 Remove the studies with duplicate or unreadable source files
duplicate_studies = [                                                                 ### List of duplicate (same content appears under multiple study names)
    "Tituler, Höftberger - Dalmau, 2014\n\nKruer, 2010",
    "Tituler, Höftberger - Dalmau, 2014\n\nYamamoto, 2013\n\nKokubun, 2016",
    "Tituler, Höftberger - Dalmau, 2014\n\nSakamoto, 2013",
    "Prüss-Wandinger, 2010 and Finke, 2012"
]
unreadable_file = ["Raynor -Berkowitz, 2016"]                                         ### List of studies linked to unreadable files

for k in duplicate_studies + unreadable_file:
    if k in study_to_text:
        del study_to_text[k]
        print(f" Removed: {repr(k)}")
    else:
        print(f"⚠️ Not found: {repr(k)}")                                             ### Warn if the key isn't found

print(f" Total number of studies in the final dataset: {len(study_to_text)}")         ### Print final number of valid studies retained for dataset split


# In[21]:


# 2.4 Save the extracted and structured text data as JSON
with open(os.path.join(output_dir, 'study_to_text.json'), 'w', encoding='utf-8') as f:
    json.dump(study_to_text, f, ensure_ascii=False, indent=2)


# In[23]:


# 3.1 Load the CSV file containing the number of cases per study and standardize the study name
cases_df = pd.read_csv(CSV_path)
cases_df["Study Name"] = cases_df["0"].astype(str)


# In[25]:


# 3.2 Processes on the cases-per-study data (merge the duplicate and delete the unreadable)

# 3.2.1 Map the duplicate study names to be merged  
merge_mapping = {
    "Tituler, Höftberger - Dalmau, 2014": [
        "Tituler, Höftberger - Dalmau, 2014",
        "Tituler, Höftberger - Dalmau, 2014\n\nKruer, 2010",
        "Tituler, Höftberger - Dalmau, 2014\n\nYamamoto, 2013\n\nKokubun, 2016",
        "Tituler, Höftberger - Dalmau, 2014\n\nSakamoto, 2013"
    ],
    "Prüss-Wandinger, 2010": [
        "Prüss-Wandinger, 2010",
        "Prüss-Wandinger, 2010 and Finke, 2012"
    ]
}

# 3.2.2 Build a list of new merged rows with summed counts
merged_rows = []
for new_name, original_names in merge_mapping.items():
    sub_df = cases_df[cases_df["Study Name"].isin(original_names)]
    total_count = sub_df["count"].sum()
    merged_rows.append({
        "Study Name": new_name,
        "count": total_count
    })

merged_df = pd.DataFrame(merged_rows)
print(" The merged studies and their sum of case number:")
print(merged_df)

# 3.2.3 Remove the original rows and append the merged rows back
cases_df = cases_df[~cases_df["Study Name"].isin(sum(merge_mapping.values(), []))]
cases_df = pd.concat([cases_df, merged_df], ignore_index=True)

# 3.2.4 Remove row associated with unreadable file
cases_df = cases_df[cases_df["Study Name"] != "Raynor -Berkowitz, 2016"]

# 3.2.5 Check the final number of studies in cases-per-study data
print(f"\n Final dataset size: {len(cases_df)}")


# In[27]:


# 3.3 Categorize studies by the number of cases
one_case_reports = cases_df[cases_df["count"] == 1]["Study Name"].tolist()
multi_case_series = cases_df[cases_df["count"] > 1]["Study Name"].tolist()

print(f" Number of case reports with only one case: {len(one_case_reports)}")                    ### Print counts of each category
print(f" Number of case series with multiple cases: {len(multi_case_series)}")


# In[29]:


# 3.4 Randomly sample studies for pilot and development sets
random.seed(42)

pilot_sample = random.sample(one_case_reports, 5)
remaining_one_case = list(set(one_case_reports) - set(pilot_sample))             ### Exclude the pilot sample from the rest of the single-case reports
development_one_case_sample = random.sample(remaining_one_case, 20)
development_multi_cases_sample = random.sample(multi_case_series, 10)


# In[31]:


# 3.5 Construct pilot, development, and validation sets from study-text dictionary
pilot_set = {k: study_to_text[k] for k in pilot_sample}

development_set = {k: study_to_text[k]
                   for k in (development_one_case_sample + development_multi_cases_sample)}

all_selected_keys = set(pilot_sample + development_one_case_sample + development_multi_cases_sample)        ### Include all remaining studies as the validation set
validation_set = {k: v for k, v in study_to_text.items() if k not in all_selected_keys}


# In[33]:


# 3.6 Save the three datasets as JSON
with open(os.path.join(output_dir, 'pilot_set.json'), 'w', encoding='utf-8') as f:
    json.dump(pilot_set, f, ensure_ascii=False, indent=2)

with open(os.path.join(output_dir, 'development_set.json'), 'w', encoding='utf-8') as f:
    json.dump(development_set, f, ensure_ascii=False, indent=2)

with open(os.path.join(output_dir, 'validation_set.json'), 'w', encoding='utf-8') as f:
    json.dump(validation_set, f, ensure_ascii=False, indent=2)

