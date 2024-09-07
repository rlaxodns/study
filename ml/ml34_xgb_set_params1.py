from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=4123,
                                                    # stratify=y
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators':100,
    'learnin_rate':0.1,
    'max_depth':5
}

#2. model
model = XGBRFRegressor(**parameters
                       ,random_state = 777)

model.set_params(gamma = 0.4)


#3.훈련
model.fit(x_train, y_train)

#4. 평가 및 예측
result = model.score(x_test, y_test)
print(result)
print(model.set_params)