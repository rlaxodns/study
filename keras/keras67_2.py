import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, Bidirectional, Input,BatchNormalization
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
train1['Standard Type'] = train1['Standard Type'].apply(smiles_to_fingerprint)
train1['Standard Relation'] = train1['Standard Relation'].apply(smiles_to_fingerprint)
# train1['Standard Value'] = train1['Standard Value'].apply(smiles_to_fingerprint)
train1['Standard Units'] = train1['Standard Units'].apply(smiles_to_fingerprint)
# train1['pChEMBL Value'] = train1['pChEMBL Value'].apply(smiles_to_fingerprint)
train1['Assay ChEMBL ID'] = train1['Assay ChEMBL ID'].apply(smiles_to_fingerprint)
train1['Target ChEMBL ID'] = train1['Target ChEMBL ID'].apply(smiles_to_fingerprint)
train1['Target Name'] = train1['Target Name'].apply(smiles_to_fingerprint)
train1['Target Organism'] = train1['Target Organism'].apply(smiles_to_fingerprint)
train1['Target Type'] = train1['Target Type'].apply(smiles_to_fingerprint)
train1['Document ChEMBL ID'] = train1['Document ChEMBL ID'].apply(smiles_to_fingerprint)
# train1['IC50_nM'] = train1['IC50_nM'].apply(smiles_to_fingerprint)


train_x = np.stack(train['Fingerprint'].values) # 1952, 2048
train_x2 = np.stack(train1['Molecule ChEMBL ID'].values)
train_x2 = np.stack(train1['Standard Type'].values)
train_x2 = np.stack(train1['Standard Relation'].values)
train_x2 = np.stack(train1['Standard Units'].values)
train_x2 = np.stack(train1['Assay ChEMBL ID'].values)
train_x2 = np.stack(train1['Target ChEMBL ID'].values)
train_x2 = np.stack(train1['Target Name'].values)
train_x2 = np.stack(train1['Target Type'].values)
train_x2 = np.stack(train1['Target Organism'].values)
train_x2 = np.stack(train1['Document ChEMBL ID'].values)
train_y = train['pIC50'].values

# print(len(np.unique(train_x2)))

# 학습 및 검증 데이터 분리
x1_train, x1_test, y_train, y_test = train_test_split(train_x, train_y, 
                                                                         test_size=0.2,
                                                                         random_state=6265)

# print(x1_train.shape, x2_train.shape, y_train.shape) #(1561, 2048) (1561, 2048) (1561,)
# print(x1_test.shape, x2_test.shape, y_test.shape) #(391, 2048) (391, 2048) (391,)


# 모델 구성
Input1 = Input(shape=(2048,))
m1 = Embedding(2048, 1024)(Input1)
m1 = Conv1D(512, 2, activation='relu')(m1)
m1 = Conv1D(512, 2, activation='relu')(m1)
m1 = Bidirectional(LSTM(256, activation='relu'))(m1)
m1 = Dense(128, activation='relu')(m1)
m1 = Dense(64, activation='relu')(m1)

Input2 = Input(shape=(2048,))
m2 = Embedding(2048, 1024)(Input2)
m2 = Conv1D(512, 2, activation='relu')(m2)
m2 = Conv1D(512, 2, activation='relu')(m2)
m2 = Bidirectional(LSTM(256, activation='relu'))(m2)
m2 = Dense(128, activation='relu')(m2)
m2 = Dense(64, activation='relu')(m2)

from keras.layers.merge import concatenate
merge = concatenate([m1, m2])
merge = Dense(128, activation='relu')(merge)
# merge = Dropout(0.2)(merge)
merge = BatchNormalization()(merge)
merge = Dense(64, activation='relu')(merge)
merge = Dense(32, activation='relu')(merge)
# merge = Dropout(0.2)(merge)
merge = Dense(1)(merge)

model = Model(inputs = [Input1, Input2], outputs = [m1, m2])



# # 토큰나이져
# from tensorflow.keras.preprocessing.text import Tokenizer
# token = Tokenizer()
# token.fit_on_texts(train_x)
# train_x = token.texts_to_sequences(train_x)

# print(train_x)
model.compile(loss= 'mse', optimizer='adam', metrics=['acc'])
model.fit([x1_train, x2_train], y_train,
          epochs=1,
          batch_size=512,
          validation_split=0.3)


test = pd.read_csv(path + 'test.csv')
test['Fingerprint'] = test['Smiles'].apply(smiles_to_fingerprint)

test_x = np.stack(test['Fingerprint'].values)

# Validation 데이터로부터의 학습 모델 평가
loss = model.evaluate([x1_test, x2_test], y_test)
result = model.predict(test_x)



##


test_y_pred = model.predict(test_x)

submit = pd.read_csv(path+'sample_submission.csv')
submit['IC50_nM'] = result
submit.head()

submit.to_csv('./baseline_submit.csv', index=False)