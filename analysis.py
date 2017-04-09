import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import OrderedDict
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

print 'lel'

DATADIR = 'C://Users//Yordanka//Documents//Paspaldzhiev//random//temps_mr//data_cleaned'

test = pd.read_csv(os.path.join(DATADIR, 'giss_anom.csv'))

predict_todo = OrderedDict()
	
year_begin = 1979
year_end = 2011 # last year in AOD data

for each in os.listdir(DATADIR):
	if 'giss' in each:
		continue
	datakey = each.split('_')[1].split('.')[0]
	data = pd.read_csv(os.path.join(DATADIR,each))
	sub = np.logical_and(year_begin <= data['year'], data['year'] <= year_end)
	if datakey == 'mei':
		ts = data[sub][data.keys()[1:]].as_matrix().flatten()
		predict_todo[datakey] = ts
	elif datakey == 'sol':
		ts = data[sub].groupby(['year','month']).mean().as_matrix()
		predict_todo[datakey] = ts
	elif datakey == 'aod':
		ts = data[sub]['val'].as_matrix()
		predict_todo[datakey] = ts
	elif datakey == 'co2':
		ts = data[sub]['val'].as_matrix()
		predict_todo[datakey] = ts
				
test_use = test[np.logical_and(year_begin <= test['year'], test['year'] <= year_end)][test.keys()[1:]].as_matrix().flatten()

predict_todo['anom'] = test_use
predict_todo['time'] = pd.date_range(str(year_begin),str(year_end+1),freq='M').values

for k,v in predict_todo.iteritems():
	predict_todo[k] = np.squeeze(v)
	
df = pd.DataFrame(predict_todo, columns = predict_todo.keys())
df['time'] = np.arange(0,len(df['time'].values))
y = df['anom']
x = np.column_stack([df['time'], df['mei'], df['sol'], df['aod'], df['co2']])
x = sm.add_constant(x, prepend=True)
model = sm.OLS(y,x).fit()
#model = ols('anom~time+sol+aod+mei', data).fit()
params = model.params
print model.summary()	
print params
cleaned = y - params[2]*df['mei'] - params[3]*df['sol']  - params[4]*df['aod'] - params[5]*df['co2']

yanom = np.average(y.reshape(-1,12),axis=1)
yclean = np.average(cleaned.reshape(-1,12),axis=1)

plt.figure()
plt.plot(df['time'],df['anom'],label='anom')
plt.plot(df['time'],cleaned,label='clean')
plt.legend()
plt.figure()
plt.plot(np.arange(year_begin,year_end+1), yanom, label='anom')
plt.plot(np.arange(year_begin,year_end+1), yclean, label='clean')
plt.legend()
plt.show()
