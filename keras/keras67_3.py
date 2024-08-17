import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

import pandas as pd
import numpy as np
import os
import random

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time

st = time.time()
CFG = {
    'NBITS':2048,
    'SEED':42,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(CFG['SEED']) # Seed 고정

# SMILES 데이터를 분자 지문으로 변환
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CFG['NBITS'])
        return np.array(fp)
    else:
        return np.zeros((CFG['NBITS'],))
    


CFG = {
    'NBITS':2048,
    'SEED':42,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(CFG['SEED']) # Seed 고정

# SMILES 데이터를 분자 지문으로 변환
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CFG['NBITS'])
        return np.array(fp)
    else:
        return np.zeros((CFG['NBITS'],))


# 학습 ChEMBL 데이터 로드
path = "C:\\ai5\\_data\\dacon\\신약개발\\"
chembl_data = pd.read_csv(path +'train.csv')  # 예시 파일 이름
chembl_data.head()
print(chembl_data.shape) #(1952,15)


train = chembl_data[['Smiles', 'pIC50']]
train1 = chembl_data.drop(['Smiles', 'pIC50'], axis=1)
print(train1.columns)
# Index(['Molecule ChEMBL ID', 'Standard Type', 'Standard Relation',
#        'Standard Value', 'Standard Units', 'pChEMBL Value', 'Assay ChEMBL ID',
#        'Target ChEMBL ID', 'Target Name', 'Target Organism', 'Target Type',
#        'Document ChEMBL ID', 'IC50_nM'],
#       dtype='object')

train['Fingerprint'] = train['Smiles'].apply(smiles_to_fingerprint)
train1['Molecule ChEMBL ID'] = train1['Molecule ChEMBL ID'].apply(smiles_to_fingerprint)

train_x = np.stack(train['Fingerprint'].values) # 1952, 2048
train_y = train['pIC50'].values

# 학습 및 검증 데이터 분리
x1_train, x1_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2,
                                                    random_state=6265)

model = Sequential()
model.add(Embedding(2048, 512))
model.add(Conv1D(256, 2))
model.add(Conv1D(256, 2))
model.add(Conv1D(256, 2))
model.add(Bidirectional(LSTM(256)))
model.add(Dense(126))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

from keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
    verbose=1,

)
model.compile(loss = 'mse', optimizer='adam')
model.fit(x1_train, y_train,
          epochs = 1000, 
          batch_size=16,
          validation_split=0.3,
          callbacks=[es])



test = pd.read_csv(path + 'test.csv')
test['Fingerprint'] = test['Smiles'].apply(smiles_to_fingerprint)

test_x = np.stack(test['Fingerprint'].values)
et = time.time()

loss = model.evaluate(x1_test, y_test)
result = model.predict(test_x)

submit = pd.read_csv(path+'sample_submission.csv')
submit['IC50_nM'] = result
submit.head()

submit.to_csv('./baseline_submit2.csv', index=False)

print(loss)
print("걸린시간", et - st)

# 0) 1.0355170965194702