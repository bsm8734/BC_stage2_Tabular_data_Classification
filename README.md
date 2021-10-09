# 로그 데이터를 이용한 미래 구매액 예측 경진대회

- 부스트캠프 AI Tech 1기 과정 중, P stage 2 기간 동안 참여한 정형데이터 분류 경진대회 소스코드 입니다.
- 대회기간: `2021.04.` (2 weeks)

### 대회 설명
- 온라인 거래 고객 log 데이터를 이용하여 고객들의 미래 소비를 예측 분석하는 프로젝트입니다.
- 5914명의 `2009년 11월 ~ 2011년 11월` 데이터를 이용하여 각 고객들의 `2011년 12월`의 **총 구매액이 300을 넘을지의 확률값을 예측**하는 이진 분류 문제입니다.
- 2011년 12월 총 구매액이 300을 넘으면 1, 넘지 않으면 0으로 예측하는 문제입니다. (고객별 예측 실시)

### 결과

- ROC-AUC: `0.8601`
- 등수: **18등** `(18/96)`

### 데이터 설명

- 2009년 12월부터 2011년 11월까지의 온라인 상점의 거래 데이터가 주어짐
- 2011년 11월 까지 데이터를 이용하여 2011년 12월의 고객 구매액 300초과 여부를 예측해야 함
- **Unique Customer_id** : 5914명
- **Customer 당 로그 수** : 1개 ~ 12714개

### 데이터 컬럼 설명

- order_id : 주문 번호. 데이터에서 같은 주문번호는 동일 주문을 나타냄
- product_id : 상품 번호
- description : 상품 설명
- quantity : 상품 주문 수량
- order_date : 주문 일자
- price : 상품 가격
- customer_id : 고객 번호
- country : 고객 거주 국가
- total : 총 구매액(quantity X price)

### 평가방식

- AUC(Area Under Curve)

### 사용한 아키텍처

- 사용된 ML 알고리즘: **LightGBM**
- 하이퍼파라미터
    ```python
    model_params = {
        'objective': 'binary', # 이진 분류
        'boosting_type': 'gbdt',
        'metric': 'auc', # 평가 지표 설정
        'feature_fraction': 0.8, # 피처 샘플링 비율
        'bagging_fraction': 0.8, # 데이터 샘플링 비율
        'bagging_freq': 1,
        'n_estimators': 10000, # 트리 개수
        'early_stopping_rounds': 100,
        'seed': SEED,
        'verbose': -1,
        'n_jobs': -1,    
    }
    ```

### Feature engineering & Other methods

| Index 	| Feature 	| Description 	| Intention 	|
|:---:|:---:|---	|---	|
| 1 	| cumsum 	| - 기존의 각 feature에 대한 누적합을 계산 	| - 현재 시점에서는 미래의 소비를 알 수 없으니, 평균이나 합 등의 aggregation function은 현재 이전의 값들에만 영향을 받아야함<br>따라서 현재 이전의 값 만을 활용한 feature가 적절할 것이라 생각하여 추가함 	|
| 2 	| order_ts 	| - 가장 최근에 구매한 total sum(last)<br>- 가장 처음에 구매한 total sum(first) 	| - 가장 최근에 구매한 총액(total)이 타켓 month에 영향을 줄 것이라고 생각함 	|
| 3 	| order_ts_plus 	| - 가장 최근에 구매한 금액 중, 양수인 값들의 total sum(last)<br>- 가장 처음에 구매한 금액 중, 양수인 값들의 total sum(first) 	| - 음수인 값들이 들어가는 것이 어떤 영향을 끼치는지 확인하고자 feature를 추가 	|
| 4 	| mode 	| - 각 feature 당 가장 많이 나온 값(최빈값)을 다시 feature로 삼음 	|  	|
| 5 	| cycle_1224 	| - 각 사용자가 1년 전(12개월 전)과 2년 전(24개월 전)에 구매한 총액의 평균을 feature로 삼음<br>- aggregation function을 적용하지 않음 	| - 매년 OO월에 300이상 구매할 확률을 알 수 있으므로, feature로서 적절하다고 생각함 	|
| 6 	| trend 	| - OO개월 전의 데이터에 대해서 customer 별로 각각 aggregation function을 적용한 결과를  feature로 삼음<br>- price, quantity, total에 대해서만 적용<br>- 대상: [1, 2, 3, 5, 7, 12, 20, 23]<br>- 기존 aggregation function을 함께 적용하는 것이 아닌, 따로 aggregation function을 적용하고, 마지막에 데이터프레임에 추가하는 형식으로 사용 	| - 장기적인 관점에서 봤을때 그래프가 증가하는지, 감소하는지, 또는 정체되어 있는지 등의 추세를 알기위해서 사용함 <br>- 그러나 이전의 데이터를 전부 다 더하는 것이 아닌, 최근 OO개월의 데이터만을 본다는 점에서 기존 feature와 다름 	|
| 7 	| seasonality 	| - 주기성을 모델이 학습할 수 있도록, 구간을 나누어 aggregation function을 적용함<br>- (1~3개월전), (6-8개월전), (12-14개월전), (18-20개월전) 이런 식으로 데이터를 묶어서 aggregation을 customer 별로 할 수 있도록 함<br>- 주기: [1, 6, 12, 18] 	| - 예측하고자 하는 12월에는 변동폭이 꽤 커서 해당 주기성을 모델이 학습하는 것 또한 중요하다고 생각함 	|

- 이외 적용한 것: **Quantile Transform**
  - not feature, 전처리
  - 데이터 스케일링을 위해서 사용
  - 변수들의 스케일을 0~1 사이로 조정하므로, 속도가 빨라진다는 장점이 있음


#### Feature importance

<img src="feature importance.png" width="80%">

<br>

> 경진대회 과정에 대한 기록, 사용한 아키텍처는 [Notion](https://shy-perfume-f1a.notion.site/Wrap-Up-2fe14d302e34431da0eed87a051ed013)에 `wrap-up report`로 올려두었습니다.
