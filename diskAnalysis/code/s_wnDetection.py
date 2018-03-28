import pandas as pd

data_file = '../data/data_processed.csv'
data = pd.read_csv(data_file)
data = data.iloc[: len(data)-5]

# stationary detection
from statsmodels.tsa.stattools import adfuller as ADF
diff =0
adf = ADF(data['used_c'])
print('orignal p is: %s'%adf[1])
while (adf[1] >= 0.05):
    diff = diff+1
    adf = ADF(data['used_c'].diff(diff).dropna())

print('%s order back to stationary, after adf p is: %s'%(diff, adf[1]))

# white noise detection 
from statsmodels.stats.diagnostic import acorr_ljungbox
[[lb], [p]] = acorr_ljungbox(data['used_c'], lags=1)
if p <0.05:
    print('no white noise, p is: %s' %p)
else:
    print('white noise, p is: %s' %p)

# 1 order diff
[[lb], [p]] = acorr_ljungbox(data['used_c'].diff().dropna(), lags=1)
if p <0.05:
    print('no white noise after 1 order diff, p is: %s' %p)
else:
    print('white noise after 1 order diff, p is: %s' %p)
