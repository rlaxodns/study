from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from xgboost import XGBRFRegressor
from sklearn.metrics import r2_score
import time

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=77) 

Random_state = 777
model1 = DecisionTreeRegressor(random_state=Random_state)
model2 = RandomForestRegressor(random_state=Random_state)
model3 = GradientBoostingRegressor(random_state=Random_state)
model4 = XGBRegressor(random_state = Random_state)


models = [model1, model2, model3, model4]


print(Random_state)
for i in models:
    i.fit(x_train, y_train)
    print('========================', i.__class__.__name__, 'Random_state = ',Random_state, "======================")
    print('acc', i.score(x_test, y_test))
    print(i.feature_importances_)
