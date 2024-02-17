"""Script to easily convert the run log to a dictionary of metrics
Simply insert the run log into the file_text variable and run the script"""
import re
import numpy as np

delta_dict = {}
bert_dict = {}
bleu_dict = {}

# Read the text file
file_text = """"""

grouped_list = [' '.join(file_text.split("\n")[i:i+3]) for i in range(0, len(file_text.split("\n")), 3)]

for text in grouped_list:
    # Define the regular expressions
    regex_file = r'file: (.*).txt'
    regex_delta = r'delta: ([-0-9.]+)'
    regex_target = r'target: ([-0-9.]+)'
    regex_error = r'error: ([-0-9.]+)'
    regex_percent_error = r'percent error: ([-0-9.]+)'
    regex_bertF1 = r'bertF1: ([-0-9.]+)'
    regex_sim_score = r'sim_score: ([-0-9.]+)'
    
    # Extract the values using regular expressions
    feature, target_delta = re.search(regex_file, text).group(1).replace('generations/', '').split("_")
    delta = float(re.search(regex_delta, text).group(1))
    target = float(re.search(regex_target, text).group(1))
    error = float(re.search(regex_error, text).group(1))
    percent_error = float(re.search(regex_percent_error, text).group(1))
    bertF1 = float(re.search(regex_bertF1, text).group(1))
    sim_score = float(re.search(regex_sim_score, text).group(1))
    
    if feature  not in delta_dict:
        delta_dict[feature] = {}
        bert_dict[feature] = {}
        bleu_dict[feature] = {}
    
    delta_dict[feature][target_delta] = [round(delta, 4)]
    bert_dict[feature][target_delta] = [bertF1]
    bleu_dict[feature][target_delta] = [round(sim_score, 4)]

# Print the resulting dictionary
print(delta_dict)
print(bert_dict)
print(bleu_dict)