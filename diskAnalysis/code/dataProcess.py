import pandas as pd
import numpy as np

discfile = '../data/discdata.xls' 
transformeddata = '../data/data_processed.csv' 
data = pd.read_excel(discfile)
data = data[data['TARGET_ID'] == 184].copy()  

data_group = data.groupby('COLLECTTIME')  # group by time

def attr_trans(x):
  result = pd.Series(index = ['name','used_c','used_d','collect_time'])
  result['name'] = x['NAME'].iloc[0]
  result['collect_time'] = x['COLLECTTIME'].iloc[0]
  result['used_c'] = x['VALUE'].iloc[0]
  result['used_d'] = x['VALUE'].iloc[1]
  return result

data_processed = data_group.apply(attr_trans) 
# data_processed.to_csv(transformeddata, index = False)

length = range(0, len(data_processed))

## plot data
import matplotlib.pyplot as plt

plt.figure()
plt.plot(length, np.array(data_processed['used_c'].copy()),'go-',label='used_c')
plt.plot(length, np.array(data_processed['used_d'].copy()),'bo-',label='used_d')
plt.xlabel('Day')
plt.ylabel('Used Information')
plt.legend(loc='upper left')
plt.show()
