import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
import math

data_file = '../data/data_processed.csv'
data = pd.read_csv(data_file, index_col='collect_time')
# data = data.iloc[: len(data)-5]
# xdata = data['used_c']
'''
# model identification
pmax = int(len(xdata)/10)
qmax = int(len(xdata)/10)
bic_matrix = []
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):
        try:
            tmp.append(ARIMA(xdata, (p,1,q)).fit().bic)
        except:
            tmp.append(None)

    bic_matrix.append(tmp)

bic_matrix = pd.DataFrame(bic_matrix)
print(bic_matrix)
p, q = bic_matrix.stack().idxmin()
print('BIC minimize p,q is: %s %s'%(p,q))

# model detection
arima = ARIMA(xdata, (0, 1, 0)).fit()
xdata_pred = arima.predict(typ='levels')
pred_error = (xdata_pred - xdata).dropna()


lb, p = acorr_ljungbox(pred_error, lags=12)
h = (p<0.05).sum()
if h >0:
    print('(p,d,q) is not white noise')
else:
    print('(p,d,q) is white noise')

'''
# test (p,d,q) model
test_data = data.iloc[len(data)-5:len(data)+1]
real_data_GB = test_data['used_c']/(1024*1024)
arima = ARIMA(real_data_GB, (0, 1, 0)).fit()
pred_data_GB = arima.predict(typ='levels')
print('real_data_GB is:')
print(real_data_GB)
print('pred_data_GB is:')
print(pred_data_GB)

# model evalute
abs_ = (pred_data_GB-real_data_GB).abs()
mae_ = abs_.mean()
rmse_ = ((abs_**2).mean())**0.5
mape_ = (abs_/real_data_GB).mean()
print('mae: %0.4f, rmse: %0.4f, mape: %0.4f'%(mae_, rmse_, mape_))

