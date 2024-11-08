"""
결측치 처리
1. 삭제 = 행 또는 열
2. 임의의 값
    평균 : mean(이상치의 문제가 존재)
    중위값: medium
    최빈값: mode
    0: fillna(0)
    앞값: ffill
    뒷값: bfill
    특정값: 임의값 (무언가 조건을 추가하는게 좋다)
3. 보간법(interpolate): 둘 사이의 변숫 삾에 대한 함수값 내지 근사값 채우기
4. 모델: .predict (회귀를 통해서 채우기)
5. 부스팅 계열 모델: 이상치와 결측치 처리가 필요없음
"""
import pandas as pd
import numpy as np

dates = ['10/11/2024', '10/12/2024', '10/13/2024',
         '10/14/2024', '10/15/2024', '10/16/2024']

dates = pd.to_datetime(dates)
print(dates)

print('===================================')
ts = pd.Series([2, np.nan, np.nan, 8, 10, np.nan], index= dates)
print(ts)

ts = ts.interpolate()
print(ts)