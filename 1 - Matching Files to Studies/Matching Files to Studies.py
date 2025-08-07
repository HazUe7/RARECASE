#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 0.1 Import required libraries
import pandas as pd
import os
import re
import shutil
from pathlib import Path
from collections import defaultdict


# In[3]:


# 0.2 Set path variables
CSV_path = 'Cases per paper.csv'          ### CSV file containing the identifiers of included studies and the numbers of cases per study
files_dir = 'publication_files'           ### Folder containing all raw publication files
output_dir = 'matching_outputs'           ### Directory to save the intermediate matching outputs
input_dir = 'manual_edited'               ### Folder containing manually checked or edited files for further processing


# In[5]:


# 1.1 Define a function to set study keys according to filenames
def extract_study_key(filename):
    """
    Extract a standardized study key from a filename.
    Parameters:
        filename (str): The filename string to parse, expected to start with a year followed by the first author's name
    Returns:
        str or None: A string in the format "YYYY[.MM], Author" if matched, else None.
    """
    pattern = r"^(\d{4}(?:\.(0?[1-9]|1[0-2]))?),\s*([A-Za-z]+)"            ### Regular expression pattern to match
    match = re.match(pattern, filename)
    if match:
        year_month = match.group(1)
        first_author = match.group(3)
        return f"{year_month}, {first_author}"                             ### Extract year/month and author's last name
    else:
        return None


# In[7]:


# 1.2 Group files by their extracted study key
study_groups = defaultdict(list)
for filename in os.listdir(files_dir):
    filepath = os.path.join(files_dir, filename)
    if os.path.isfile(filepath):
        key = extract_study_key(filename)               ### Extract a unique study key from the filename
        study_groups[key].append(filepath)              ### Group the file path under the extracted key


# In[9]:


# 1.3 Structure and save the data of grouped files for further manual process
grouped_data = []
for key, files in study_groups.items():
    for file in files:                                              ### For each file, store the study key, file name, and full file path
        grouped_data.append({
            "Study Group Key": key,
            "Filename": os.path.basename(file),
            "Full Path": file
        })

os.makedirs(output_dir, exist_ok=True)
to_edit_output = os.path.join(output_dir, "study_groups_review.xlsx")

df = pd.DataFrame(grouped_data)
prefix = "publication_files"
df["Full Path"] = df["Full Path"].str.replace(prefix, "NMDARE SR articles", regex=False)         ### Filepath for manually processing folder

df.to_excel(to_edit_output, index=False)                               ### Export the DataFrame to an Excel file without the row index
print(f"Output file for further manual process: {to_edit_output}")


# In[ ]:


### Manually delete duplicate files and separate different studies into different groups in the saved Excel file.
### The intermediate outputs of manual process are the Excel file "study_groups_review_edited" and a folder named "NMDARE SR articles - intermediate".


# In[11]:


# 2.1 Read the manually processed Excel file into a dictionary of lists to regroup file paths by study key
excel_path = os.path.join(input_dir, "study_groups_review_edited.xlsx")
df = pd.read_excel(excel_path)

study_groups = defaultdict(list)

for _, row in df.iterrows():                          # Extract and clean the study group key and the full file path
    key = str(row["Study Group Key"]).strip()
    filepath = str(row["Full Path"]).strip()

    if key and filepath:
        study_groups[key].append(filepath)


# In[13]:


# 2.2 Summarize the total number of unique study groups and files across all study groups
total_groups = len(study_groups)
total_files = sum(len(files) for files in study_groups.values())

print(f"Total study groups: {total_groups}")
print(f"Total files across all groups: {total_files}")


# In[15]:


# 3.1 Load the list of included studies
df = pd.read_csv(CSV_path)
study_names = df.iloc[:, 0].dropna().astype(str).tolist()


# In[17]:


# 3.2 Convert the keys of study groups to a list
study_group_names = pd.Series(study_groups.keys()).astype(str).tolist()


# In[23]:


# 3.3 Define functions for preprocessing the study names and group names

# 3.3.1 Define a fuction to split a study/group name into parts
def split_study_name(study):
    """
    Split string into parts based on delimiters.
    Parameters:
        study (str): The study/group name to be split.
    Returns:
        list[str]: A list of non-empty string parts obtained by splitting the input string.
    """
    parts = re.split(r'[ .,\-׳]+', study)
    parts = [p for p in parts if p]
    return parts

# 3.3.2 Define a function to identify the publication year.
def is_year(part):
    """
    Check if a given string part represents a year in the 4-digit format.
    Parameters:
        part (str): The string part to check.
    Returns:
        bool: True if the part matches exactly 4 digits, False otherwise.
    """
    return bool(re.fullmatch(r'\d{4}', part))

# 3.3.3 Define a function to classify the components of a study name into years and authors
def analyze_study_parts(study):
    """
    Split a name string and classify parts into years and authors.
    Parameters:
        study (str): The study name string to analyze.
    Returns:
        tuple:
            - list[str]: A list of regex patterns matching the year if exist; otherwise empty list.
            - list[str]: A list of regex patterns matching the author names as whole words.
    """
    parts = split_study_name(study)
    years = [p for p in parts if is_year(p)]
    authors = [p for p in parts if not is_year(p)]
    
    author_patterns = [r'\b' + re.escape(a) + r'\b' for a in authors]       ### Create regex patterns to match the identified years and authors as whole words.
    if len(years) == 1:
        year_patterns = [r'\b' + years[0] + r'\b']
    else:
        year_patterns = []
        
    return year_patterns, author_patterns


# In[25]:


# 3.4 Define the function for matching the study groups to the target study names
def match_study_to_file(study, group_name):
    """
    Determine whether a group name matches a target study name based on author and year components.
    The function compares the authors and years extracted from both study and group names, based on the following matching rules.
        - All authors in the group name must be present in the study name.
        - If years are identified in the study name, the years must exactly match those in the group name.
    Parameters:
        study (str): The study name string to be matched.
        group_name (str): The group name string to match against.
    Returns:
        bool: True if the study and the group name matches; False otherwise.
    """
    study_years, study_authors = analyze_study_parts(study)
    group_years, group_authors = analyze_study_parts(group_name)

    for ap in group_authors:                          ### All authors in the group name must be present in the study name
        if ap not in study_authors:
            return False

    if study_years:                                   ### If years are identified in the study name, the years must exactly match those in the group name
        if not (study_years == group_years):
            return False
    
    return True


# In[27]:


# 3.5 Conduct matching through all group names for each study name
matched = defaultdict(list)                            ### Initialize a dictionary for lists of matched group names for each study name

for study in study_names:
    for group_name in study_group_names:
        if match_study_to_file(study, group_name):
            matched[study].append(group_name)          ### Append the group name to the list of the processing study name if a match is found 


# In[31]:


# 3.6 Save the data of matched study-group pairs for further manual validation and edition
matched_df = pd.DataFrame(
    [(study, match) for study, matches in matched.items() for match in matches],
    columns=['Study Name', 'Matched Group Name']
)

matched_output = os.path.join(output_dir, "matched_output.xlsx")
matched_df.to_excel(matched_output, index=False)


# In[33]:


# 3.7 Summarize the unmatched study names and group names and save them for further manual validation and edition

# 3.7.1 Identify study names that did not match any group
matched_study_names = set(matched.keys())
unmatched_study_names = [s for s in study_names if s not in matched_study_names]

# 3.7.2 Identify group names that were not matched to any study
matched_group_names = set(group_name for matched_list in matched.values() for group_name in matched_list)
unmatched_group_names = [g for g in study_group_names if g not in matched_group_names]

# 3.7.3 Compute the numbers of unmatched studies and groups
print(f"Number of Unmatched Studies:{len(unmatched_study_names)}")
print(f"Number of Unmatched Groups:{len(unmatched_group_names)}")

# 3.7.4 Save the data of unmatched study names
unmatched_studies_df = pd.DataFrame(unmatched_study_names, columns=['Unmatched Study Name'])
unmatched_studies = os.path.join(output_dir, "unmatched_studies.xlsx")
unmatched_studies_df.to_excel(unmatched_studies, index=False)

# 3.7.4 Save the data of unmatched group names
unmatched_groups_df = pd.DataFrame(unmatched_group_names, columns=['Unmatched Group Name'])
unmatched_groups = os.path.join(output_dir, "unmatched_groups.xlsx")
unmatched_groups_df.to_excel(unmatched_groups, index=False)


# In[ ]:


### Manually remove duplicate entries in the matched_output and match groups for unmatched study names. Missing articles are manually added into the folder.
### The final outputs of manual process are: an Excel named "study_groups_review_final" recording all valid path to the original files and their study group names; another Excel named "matched_final" recording all 652 included studies and the study group name matched to them, and a folder named "NMDARE SR articles" containing all the corresponding original publication files.  

