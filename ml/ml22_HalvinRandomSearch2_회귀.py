from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from xgboost import XGBRFRegressor
from sklearn.metrics import r2_score
import time

import warnings
warnings.filterwarnings('ignore')

x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=77) 

kfold = KFold(
    n_splits=5,
    random_state=77,
    shuffle=True
)

parameter = [
    {'learning_rate' :[0.01, 0.05, 0.1, 0.2, 0.5], 'max_depth':[3,4,5,6,8], 
    'learning_rate' :[0.01, 0.05, 0.1, 0.2, 0.3], 'subsample':[0.6, 0.7,0.8, 0.9, 1.0],
    'learning_rate' :[0.01, 0.05, 0.1, 0.2, 0.3], 'colsample_bytree':[0.6,0.7,0.8,0.9,1.0],
    'learning_rate' :[0.01, 0.05, 0.1, 0.2, 0.3], 'gamma':[0,0.1,0.2, 0.3,0.5,1.0]}    
    ]

model = HalvingRandomSearchCV(XGBRFRegressor(
    tree_method = 'hist',
    device = 'cuda',
    n_estimators = 100
), 
                            parameter,
                            refit=True,
                            cv = kfold,
                            verbose=1,
                            random_state=77,
                            factor=4,
                            n_iter = 0
                            )

st = time.time()

model.fit(x_train, y_train)

et = time.time()

print("최적 매개변수", model.best_estimator_)
print("최적 파라미터", model.best_params_)
print("최고점수", model.best_score_)
print(model.score(x_test, y_test))

y_pre = model.predict(x_test)
print("r2", r2_score(y_test, y_pre))

y_best_pre = model.best_estimator_.predict(x_test)
print("best r2", r2_score(y_test, y_best_pre))
