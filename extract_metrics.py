'''
Author: Ashwin Kalyan
Date: 2025-10-20
Organization: Computational Biology @ Berkeley

This file automates/generalizes the computation of:
    - Population counts and percentages by ancestry group and superpopulation
    - Graphing the population data
    - Shannon Diversity Index
'''

import pandas as pd
import matplotlib.pyplot as plt
import math

population_col_title = ''
num_indiv_col_title = ''
percent_col_title = ''
superpopulation_col_title = ''
super_num_indiv_col_title = ''
super_percent_col_title = ''

path = ''

def set_path(in_path):
    path = in_path

def set_column_titles(in_population_col_title, in_num_indiv_col_title, in_percent_col_title, in_superpopulation_col_title, in_super_num_indiv_col_title, in_super_percent_col_title):
    '''
    Different data sets can have different names for the same concepts displayed in the columns. 
    This function generalizes the columns (that we need) and allows the user to input what its called in their dataset.
    '''
    population_col_title = in_population_col_title
    num_indiv_col_title = in_num_indiv_col_title
    percent_col_title = in_percent_col_title
    superpopulation_col_title = in_superpopulation_col_title
    super_num_indiv_col_title = in_super_num_indiv_col_title
    super_percent_col_title = in_super_percent_col_title
    return population_col_title, num_indiv_col_title, percent_col_title, superpopulation_col_title, super_num_indiv_col_title, super_percent_col_title

COLUMNS_GENERALIZED = {
    'population': population_col_title,
    'num_indiv': num_indiv_col_title,
    'percent': percent_col_title,
    'super_pop': superpopulation_col_title,
    'super_num_indiv': super_num_indiv_col_title,
    'super_percent': super_percent_col_title
}

# frank's code generalized
def read_data(file_path):
    """Reads data from a CSV file into a pandas DataFrame."""
    data = pd.read_csv(file_path)

    dict= {}
    for pop, percet in zip(data['super_pop'], data['super_percent']):
        if pop in dict:
            continue
        else:
            dict[pop] = percet
    
    return dict

def read_pop(file_path, pop):

    data = pd.read_csv(file_path)
    data = data[data['super_pop'] == pop]
    #print(data)
    dict= {}
    for pop, count in zip(data['population'], data['percent']):
        if pop in dict:
            continue
        else:
            dict[pop] = count
    
    return dict


plt.figure(figsize=(6, 6))
data = read_data(path)
plt.pie(
    data.values(),
    labels=data.keys(),
    autopct='%1.1f%%',    
    startangle=90,        
    counterclock=False    
)
plt.title('Ancestry Composition')

plt.savefig('ancestry_composition.png')

plt.figure(figsize=(12, 8))
for i, pop in enumerate(data.keys()):
    plt.subplot(2, 3, i+1)
    data = read_pop(path, pop)
    plt.pie(
        data.values(),
        labels=data.keys(),
        autopct='%1.1f%%',    
        startangle=90,        
        counterclock=False    
    )
    plt.title(f'{pop} Composition')
    plt.axis('equal')
plt.show()
plt.savefig('detailed_ancestry_composition.png')  

# cyan's code generalized
df = pd.read_csv(path)

shannon_sub = -sum(row * math.log(row) for row in df["percent"])
shannon_super = -sum(row * math.log(row) for row in df["super_percent"])

print(f" Shannon sub: {shannon_sub}, Shannon super: {shannon_super}")