import pandas as pd
import matplotlib.pyplot as plt

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
    for pop, count in zip(data['Population'], data['percent']):
        if pop in dict:
            continue
        else:
            dict[pop] = count
    
    return dict


plt.figure(figsize=(6, 6))
data = read_data('population_summary.csv')
plt.pie(
    data.values(),
    labels=data.keys(),
    autopct='%1.1f%%',    
    startangle=90,        
    counterclock=False    
)
plt.title('Ancestry Composition')
#plt.axis('equal')  # 保证图像为圆形
plt.savefig('ancestry_composition.png')

plt.figure(figsize=(16, 12))
for i, pop in enumerate(data.keys()):
    plt.subplot(2, 3, i+1)
    data = read_pop('population_summary.csv', pop)
    plt.pie(
        data.values(),
        labels=data.keys(),
        autopct='%1.1f%%',    
        startangle=90,        
        counterclock=False    
    )
    plt.title(f'{pop} Composition')
    plt.axis('equal')  # 保证图像为圆形
plt.savefig('detailed_composition.png')


