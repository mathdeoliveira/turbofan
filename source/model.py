import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

### LEITURA DOS ARQUIVOS

DIR = '/home/matheus/Documentos/turbofan'
DATA_DIR = os.path.join(DIR, 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train_FD001.txt')
RUL_FILE = os.path.join(DATA_DIR, 'RUL_FD001.txt')

df_train = pd.read_csv(TRAIN_FILE, header = None, delimiter= ' ')
df_rul = pd.read_csv(RUL_FILE, header = None)

## PRE PROCESSAMENTO
df_train.rename( columns = {0: 'motorid',
                           1: 'time',
                           2: 'setting01',
                           3: 'setting02',
                           4: 'setting03',
                           5: 'measurement01',
                           6: 'measurement02',
                           7: 'measurement03',
                           8: 'measurement04',
                           9: 'measurement05',
                           10: 'measurement06',
                           11: 'measurement07',
                           12: 'measurement08',
                           13: 'measurement09',
                           14: 'measurement10',
                           15: 'measurement11',
                           16: 'measurement12',
                           17: 'measurement13',
                           18: 'measurement14',
                           19: 'measurement15',
                           20: 'measurement16',
                           21: 'measurement17',
                           22: 'measurement18',
                           23: 'measurement19',
                           24: 'measurement20',
                           25: 'measurement21',
                           }, inplace = True)

df_rul.reset_index(inplace = True)
df_rul.rename(columns = {0: 'RUL',
                        'index' : 'motorid'} 
                        , inplace = True)
df_rul['motorid'] += 1
df_train.drop([26,27], axis = 1, inplace = True)

# JUNTANDO OS DOIS DATAFRAMES
df = pd.merge(df_train, df_rul, how = 'left', on=['motorid'])

## MODEL BASELINE
X = df.drop(columns = ['time'])
y = df['time']

## DADOS PARA TREINO E TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

# INSTANCIANDO O MODELO
lr = LinearRegression()

# TREINANDO O MODELO COM DADOS DE TREINO
model = lr.fit(X_train, y_train)

# SCORE MODELO
print(model.score(X_test, y_test))