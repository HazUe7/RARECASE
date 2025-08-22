{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "400dbc0c-a180-4fb4-a49d-0e2b0dfe1fbc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 0.1 Import required libraries\n",
    "import pandas as pd\n",
    "import os\n",
    "import re\n",
    "import shutil\n",
    "from pathlib import Path\n",
    "from collections import defaultdict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "87fbafba-25b6-45a4-b70d-74151098f408",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 0.2 Set path variables\n",
    "CSV_path = 'Cases per paper.csv'          ### CSV file containing the identifiers of included studies and the numbers of cases per study\n",
    "files_dir = 'publication_files'           ### Folder containing all raw publication files\n",
    "output_dir = 'matching_outputs'           ### Directory to save the intermediate matching outputs\n",
    "input_dir = 'manual_edited'               ### Folder containing manually checked or edited files for further processing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "be65ceb7-46ee-419a-bc04-896b994fdaa8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.1 Define a function to set study keys according to filenames\n",
    "def extract_study_key(filename):\n",
    "    \"\"\"\n",
    "    Extract a standardized study key from a filename.\n",
    "    Parameters:\n",
    "        filename (str): The filename string to parse, expected to start with a year followed by the first author's name\n",
    "    Returns:\n",
    "        str or None: A string in the format \"YYYY[.MM], Author\" if matched, else None.\n",
    "    \"\"\"\n",
    "    pattern = r\"^(\\d{4}(?:\\.(0?[1-9]|1[0-2]))?),\\s*([A-Za-z]+)\"            ### Regular expression pattern to match\n",
    "    match = re.match(pattern, filename)\n",
    "    if match:\n",
    "        year_month = match.group(1)\n",
    "        first_author = match.group(3)\n",
    "        return f\"{year_month}, {first_author}\"                             ### Extract year/month and author's last name\n",
    "    else:\n",
    "        return None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5ce6b22d-f8ab-48ff-9db4-cc63684ab423",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.2 Group files by their extracted study key\n",
    "study_groups = defaultdict(list)\n",
    "for filename in os.listdir(files_dir):\n",
    "    filepath = os.path.join(files_dir, filename)\n",
    "    if os.path.isfile(filepath):\n",
    "        key = extract_study_key(filename)               ### Extract a unique study key from the filename\n",
    "        study_groups[key].append(filepath)              ### Group the file path under the extracted key"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "13669b15-8790-4143-9622-e8c63a751365",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output file for further manual process: matching_outputs/study_groups_review.xlsx\n"
     ]
    }
   ],
   "source": [
    "# 1.3 Structure and save the data of grouped files for further manual process\n",
    "grouped_data = []\n",
    "for key, files in study_groups.items():\n",
    "    for file in files:                                              ### For each file, store the study key, file name, and full file path\n",
    "        grouped_data.append({\n",
    "            \"Study Group Key\": key,\n",
    "            \"Filename\": os.path.basename(file),\n",
    "            \"Full Path\": file\n",
    "        })\n",
    "\n",
    "os.makedirs(output_dir, exist_ok=True)\n",
    "to_edit_output = os.path.join(output_dir, \"study_groups_review.xlsx\")\n",
    "\n",
    "df = pd.DataFrame(grouped_data)\n",
    "prefix = \"publication_files\"\n",
    "df[\"Full Path\"] = df[\"Full Path\"].str.replace(prefix, \"NMDARE SR articles\", regex=False)         ### Filepath for manually processing folder\n",
    "\n",
    "df.to_excel(to_edit_output, index=False)                               ### Export the DataFrame to an Excel file without the row index\n",
    "print(f\"Output file for further manual process: {to_edit_output}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f01e2d12-56be-4747-9a9c-90df672ca92c",
   "metadata": {},
   "outputs": [],
   "source": [
    "### Manually delete duplicate files and separate different studies into different groups in the saved Excel file.\n",
    "### The intermediate outputs of manual process are the Excel file \"study_groups_review_edited\" and a folder named \"NMDARE SR articles - intermediate\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "07cb5210-27d1-4e1e-bdbb-8dc548d0d2be",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2.1 Read the manually processed Excel file into a dictionary of lists to regroup file paths by study key\n",
    "excel_path = os.path.join(input_dir, \"study_groups_review_edited.xlsx\")\n",
    "df = pd.read_excel(excel_path)\n",
    "\n",
    "study_groups = defaultdict(list)\n",
    "\n",
    "for _, row in df.iterrows():                          # Extract and clean the study group key and the full file path\n",
    "    key = str(row[\"Study Group Key\"]).strip()\n",
    "    filepath = str(row[\"Full Path\"]).strip()\n",
    "\n",
    "    if key and filepath:\n",
    "        study_groups[key].append(filepath)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b250d279-f114-4831-b4e8-4e177756d79a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total study groups: 790\n",
      "Total files across all groups: 813\n"
     ]
    }
   ],
   "source": [
    "# 2.2 Summarize the total number of unique study groups and files across all study groups\n",
    "total_groups = len(study_groups)\n",
    "total_files = sum(len(files) for files in study_groups.values())\n",
    "\n",
    "print(f\"Total study groups: {total_groups}\")\n",
    "print(f\"Total files across all groups: {total_files}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "bae0e207-8209-4133-aad5-6e196d09ef2f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3.1 Load the list of included studies\n",
    "df = pd.read_csv(CSV_path)\n",
    "study_names = df.iloc[:, 0].dropna().astype(str).tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b432fd02-49ef-41c4-aae9-5fa07b094d74",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3.2 Convert the keys of study groups to a list\n",
    "study_group_names = pd.Series(study_groups.keys()).astype(str).tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "9918ed44-57a7-4f10-b98d-fc580043e1c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3.3 Define functions for preprocessing the study names and group names\n",
    "\n",
    "# 3.3.1 Define a fuction to split a study/group name into parts\n",
    "def split_study_name(study):\n",
    "    \"\"\"\n",
    "    Split string into parts based on delimiters.\n",
    "    Parameters:\n",
    "        study (str): The study/group name to be split.\n",
    "    Returns:\n",
    "        list[str]: A list of non-empty string parts obtained by splitting the input string.\n",
    "    \"\"\"\n",
    "    parts = re.split(r'[ .,\\-׳]+', study)\n",
    "    parts = [p for p in parts if p]\n",
    "    return parts\n",
    "\n",
    "# 3.3.2 Define a function to identify the publication year.\n",
    "def is_year(part):\n",
    "    \"\"\"\n",
    "    Check if a given string part represents a year in the 4-digit format.\n",
    "    Parameters:\n",
    "        part (str): The string part to check.\n",
    "    Returns:\n",
    "        bool: True if the part matches exactly 4 digits, False otherwise.\n",
    "    \"\"\"\n",
    "    return bool(re.fullmatch(r'\\d{4}', part))\n",
    "\n",
    "# 3.3.3 Define a function to classify the components of a study name into years and authors\n",
    "def analyze_study_parts(study):\n",
    "    \"\"\"\n",
    "    Split a name string and classify parts into years and authors.\n",
    "    Parameters:\n",
    "        study (str): The study name string to analyze.\n",
    "    Returns:\n",
    "        tuple:\n",
    "            - list[str]: A list of regex patterns matching the year if exist; otherwise empty list.\n",
    "            - list[str]: A list of regex patterns matching the author names as whole words.\n",
    "    \"\"\"\n",
    "    parts = split_study_name(study)\n",
    "    years = [p for p in parts if is_year(p)]\n",
    "    authors = [p for p in parts if not is_year(p)]\n",
    "    \n",
    "    author_patterns = [r'\\b' + re.escape(a) + r'\\b' for a in authors]       ### Create regex patterns to match the identified years and authors as whole words.\n",
    "    if len(years) == 1:\n",
    "        year_patterns = [r'\\b' + years[0] + r'\\b']\n",
    "    else:\n",
    "        year_patterns = []\n",
    "        \n",
    "    return year_patterns, author_patterns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "53bb1118-2df5-4caf-ae70-8d2522659e0b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3.4 Define the function for matching the study groups to the target study names\n",
    "def match_study_to_file(study, group_name):\n",
    "    \"\"\"\n",
    "    Determine whether a group name matches a target study name based on author and year components.\n",
    "    The function compares the authors and years extracted from both study and group names, based on the following matching rules.\n",
    "        - All authors in the group name must be present in the study name.\n",
    "        - If years are identified in the study name, the years must exactly match those in the group name.\n",
    "    Parameters:\n",
    "        study (str): The study name string to be matched.\n",
    "        group_name (str): The group name string to match against.\n",
    "    Returns:\n",
    "        bool: True if the study and the group name matches; False otherwise.\n",
    "    \"\"\"\n",
    "    study_years, study_authors = analyze_study_parts(study)\n",
    "    group_years, group_authors = analyze_study_parts(group_name)\n",
    "\n",
    "    for ap in group_authors:                          ### All authors in the group name must be present in the study name\n",
    "        if ap not in study_authors:\n",
    "            return False\n",
    "\n",
    "    if study_years:                                   ### If years are identified in the study name, the years must exactly match those in the group name\n",
    "        if not (study_years == group_years):\n",
    "            return False\n",
    "    \n",
    "    return True"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "0f126aaa-120e-41a9-aaea-c48a2c8fba0b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3.5 Conduct matching through all group names for each study name\n",
    "matched = defaultdict(list)                            ### Initialize a dictionary for lists of matched group names for each study name\n",
    "\n",
    "for study in study_names:\n",
    "    for group_name in study_group_names:\n",
    "        if match_study_to_file(study, group_name):\n",
    "            matched[study].append(group_name)          ### Append the group name to the list of the processing study name if a match is found "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "75917f68-502e-4bca-b817-75e0d2db3734",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3.6 Save the data of matched study-group pairs for further manual validation and edition\n",
    "matched_df = pd.DataFrame(\n",
    "    [(study, match) for study, matches in matched.items() for match in matches],\n",
    "    columns=['Study Name', 'Matched Group Name']\n",
    ")\n",
    "\n",
    "matched_output = os.path.join(output_dir, \"matched_output.xlsx\")\n",
    "matched_df.to_excel(matched_output, index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "dd0542bf-f54d-4622-8c21-db564c178dad",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of Unmatched Studies:87\n",
      "Number of Unmatched Groups:212\n"
     ]
    }
   ],
   "source": [
    "# 3.7 Summarize the unmatched study names and group names and save them for further manual validation and edition\n",
    "\n",
    "# 3.7.1 Identify study names that did not match any group\n",
    "matched_study_names = set(matched.keys())\n",
    "unmatched_study_names = [s for s in study_names if s not in matched_study_names]\n",
    "\n",
    "# 3.7.2 Identify group names that were not matched to any study\n",
    "matched_group_names = set(group_name for matched_list in matched.values() for group_name in matched_list)\n",
    "unmatched_group_names = [g for g in study_group_names if g not in matched_group_names]\n",
    "\n",
    "# 3.7.3 Compute the numbers of unmatched studies and groups\n",
    "print(f\"Number of Unmatched Studies:{len(unmatched_study_names)}\")\n",
    "print(f\"Number of Unmatched Groups:{len(unmatched_group_names)}\")\n",
    "\n",
    "# 3.7.4 Save the data of unmatched study names\n",
    "unmatched_studies_df = pd.DataFrame(unmatched_study_names, columns=['Unmatched Study Name'])\n",
    "unmatched_studies = os.path.join(output_dir, \"unmatched_studies.xlsx\")\n",
    "unmatched_studies_df.to_excel(unmatched_studies, index=False)\n",
    "\n",
    "# 3.7.4 Save the data of unmatched group names\n",
    "unmatched_groups_df = pd.DataFrame(unmatched_group_names, columns=['Unmatched Group Name'])\n",
    "unmatched_groups = os.path.join(output_dir, \"unmatched_groups.xlsx\")\n",
    "unmatched_groups_df.to_excel(unmatched_groups, index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6a8bf981-0a18-4092-bc6b-1e19002c819a",
   "metadata": {},
   "outputs": [],
   "source": [
    "### Manually remove duplicate entries in the matched_output and match groups for unmatched study names. Missing articles are manually added into the folder.\n",
    "### The final outputs of manual process are: an Excel named \"study_groups_review_final\" recording all valid path to the original files and their study group names; another Excel named \"matched_final\" recording all 652 included studies and the study group name matched to them, and a folder named \"NMDARE SR articles\" containing all the corresponding original publication files.  "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (nlp_env)",
   "language": "python",
   "name": "py38env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
