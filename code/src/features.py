import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Custom library
from utils import seed_everything, print_score


TOTAL_THRES = 300 # 구매액 임계값
SEED = 42 # 랜덤 시드
seed_everything(SEED) # 시드 고정

data_dir = '../input/train.csv' # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model' # os.environ['SM_MODEL_DIR']


'''
    입력인자로 받는 year_month에 대해 고객 ID별로 총 구매액이
    구매액 임계값을 넘는지 여부의 binary label을 생성하는 함수
'''
def generate_label(df, year_month, total_thres=TOTAL_THRES, print_log=False):
    df = df.copy()
    
    # year_month에 해당하는 label 데이터 생성
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    df.reset_index(drop=True, inplace=True)

    # year_month 이전 월의 고객 ID 추출
    cust = df[df['year_month']<year_month]['customer_id'].unique()
    # year_month에 해당하는 데이터 선택
    df = df[df['year_month']==year_month]
    
    # label 데이터프레임 생성
    label = pd.DataFrame({'customer_id':cust})
    label['year_month'] = year_month
    
    # year_month에 해당하는 고객 ID의 구매액의 합 계산
    grped = df.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    
    # label 데이터프레임과 merge하고 구매액 임계값을 넘었는지 여부로 label 생성
    label = label.merge(grped, on=['customer_id','year_month'], how='left')
    label['total'].fillna(0.0, inplace=True)
    label['label'] = (label['total'] > total_thres).astype(int)

    # 고객 ID로 정렬
    label = label.sort_values('customer_id').reset_index(drop=True)
    if print_log: print(f'{year_month} - final label shape: {label.shape}')
    
    return label

def feature_preprocessing(train, test, features, do_imputing=True):
    x_tr = train.copy()
    x_te = test.copy()
    
    # 범주형 피처 이름을 저장할 변수
    cate_cols = []

    # 레이블 인코딩
    for f in features:
        if x_tr[f].dtype.name == 'object': # 데이터 타입이 object(str)이면 레이블 인코딩
            cate_cols.append(f)
            le = LabelEncoder()
            # train + test 데이터를 합쳐서 레이블 인코딩 함수에 fit
            le.fit(list(x_tr[f].values) + list(x_te[f].values))
            
            # train 데이터 레이블 인코딩 변환 수행
            x_tr[f] = le.transform(list(x_tr[f].values))
            
            # test 데이터 레이블 인코딩 변환 수행
            x_te[f] = le.transform(list(x_te[f].values))

    print('categorical feature:', cate_cols)

    if do_imputing:
        # 중위값으로 결측치 채우기
        imputer = SimpleImputer(strategy='median')

        x_tr[features] = imputer.fit_transform(x_tr[features])
        x_te[features] = imputer.transform(x_te[features])
    
    return x_tr, x_te

def feature_engineering(df, year_month):
    df = df.copy()
    
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    
    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]
    
    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    
    # group by aggregation 함수 선언
    agg_func = ['mean','max','min','sum','count','std','skew']
    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_func)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for col in train_agg.columns.levels[0]:
            for stat in train_agg.columns.levels[1]:
                new_cols.append(f'{col}-{stat}')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.groupby(['customer_id']).agg(agg_func)
    test_agg.columns = new_cols
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')

    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)

    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features

def feature_engineering1(df, year_month):
    df = df.copy()

    # customer_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_cust_id'] = df.groupby(['customer_id'])['total'].cumsum()
    df['cumsum_quantity_by_cust_id'] = df.groupby(['customer_id'])['quantity'].cumsum()
    df['cumsum_price_by_cust_id'] = df.groupby(['customer_id'])['price'].cumsum()

    # product_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_prod_id'] = df.groupby(['product_id'])['total'].cumsum()
    df['cumsum_quantity_by_prod_id'] = df.groupby(['product_id'])['quantity'].cumsum()
    df['cumsum_price_by_prod_id'] = df.groupby(['product_id'])['price'].cumsum()

    # order_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_order_id'] = df.groupby(['order_id'])['total'].cumsum()
    df['cumsum_quantity_by_order_id'] = df.groupby(['order_id'])['quantity'].cumsum()
    df['cumsum_price_by_order_id'] = df.groupby(['order_id'])['price'].cumsum()

    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')

    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]

    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id', 'year_month', 'label']]
    test_label = generate_label(df, year_month)[['customer_id', 'year_month', 'label']]

    # group by aggregation 함수 선언
    agg_func = ['mean', 'max', 'min', 'sum', 'count', 'std', 'skew']
    agg_dict = {
        'quantity': agg_func,
        'price': agg_func,
        'total': agg_func,
        'cumsum_total_by_cust_id': agg_func,
        'cumsum_quantity_by_cust_id': agg_func,
        'cumsum_price_by_cust_id': agg_func,
        'cumsum_total_by_prod_id': agg_func,
        'cumsum_quantity_by_prod_id': agg_func,
        'cumsum_price_by_prod_id': agg_func,
        'cumsum_total_by_order_id': agg_func,
        'cumsum_quantity_by_order_id': agg_func,
        'cumsum_price_by_order_id': agg_func,
        'order_id': ['nunique'],
        'product_id': ['nunique'],
    }
    all_train_data = pd.DataFrame()

    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_dict)

        new_cols = []
        for col in agg_dict.keys():
            for stat in agg_dict[col]:
                if type(stat) is str:
                    new_cols.append(f'{col}-{stat}')
                else:
                    new_cols.append(f'{col}-mode')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace=True)

        train_agg['year_month'] = tr_ym

        all_train_data = all_train_data.append(train_agg)

    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns

    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.groupby(['customer_id']).agg(agg_dict)
    test_agg.columns = new_cols

    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')

    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)

    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)

    return x_tr, x_te, all_train_data['label'], features

# def get_year_month_list(df, year_month):
#     df = df.copy()
#
#     df['year_month-mode'] = df['order_date'].dt.strftime('%Y-%m')
#     dd = df.groupby(['year_month-mode', 'customer_id'])['total'].sum()
#     cust_ids = df['customer_id'].unique()
#
#     # year_month 이전 월 계산
#     bef_12_d = datetime.datetime.strptime(year_month, "%Y-%m")
#     bef_12_prev_ym = bef_12_d - dateutil.relativedelta.relativedelta(months=12)
#     bef_12_prev_ym = bef_12_prev_ym.strftime('%Y-%m')
#
#     # ddt = df[df['year_month-mode'] == bef_12_prev_ym]
#
#     first_bef = []
#     for id in cust_ids:
#         dd[:, bef_12_prev_ym]
#         # first_bef.append(dd.xs((id, bef_12_prev_ym)))
#
#     # df['cycle_month'] = pd.Series(first_bef)
#
#     print(df)

def make_time_series_data(df, Input, year_month, stand):
    # 기준을 잡습니다. 기준은 여기서 %Y-%m 입니다.
    standard = ['customer_id'] + [stand]
    data = Input.copy()
    df = df.copy()

    data[stand] = pd.to_datetime(df['order_date']).dt.strftime(stand)
    data.order_date = pd.to_datetime(data['order_date'])

    # 월단위의 틀을 만들어주고, 기준으로 aggregation을 해준 다음에 merge를 해줄 것입니다
    times = pd.date_range('2009-12-01', periods=(data.order_date.max() - data.order_date.min()).days + 1, freq='1d')
    customerid_frame = np.repeat(data.customer_id.unique(), len(times))
    date_frame = np.tile(times, len(data.customer_id.unique()))

    frame = pd.DataFrame({'customer_id': customerid_frame, 'order_date': date_frame})
    frame[stand] = pd.to_datetime(frame.order_date).dt.strftime(stand)

    # group by
    data_group = data.groupby(standard).sum().reset_index()
    frame_group = frame.groupby(standard).count().reset_index().drop(['order_date'], axis=1)

    # merge
    merge = pd.merge(frame_group, data_group, on=standard, how='left').fillna(0)
    merge = merge.rename(columns={stand: 'standard'})

    merge_test = merge[merge['standard'] == year_month].drop(columns=['standard', 'quantity', 'price']) #.drop(merge.columns.tolist() - ['customer_id', 'total'])
    return merge_test

def add_trend(df, year_month):
    df = df.copy()
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    # train과 test 데이터 생성
    train = df[df['order_date'] < prev_ym]  # 2009-12부터 2011-10 데이터 추출
    test = df[df['order_date'] < year_month]  # 2009-12부터 2011-11 데이터 추출
    train_window_ym = []
    test_window_ym = []
    for month_back in [1, 2, 3, 5, 7, 12, 20, 23]:  # 1개월, 2개월, ... 20개월, 23개월 전 year_month 파악
        train_window_ym.append((prev_ym - dateutil.relativedelta.relativedelta(months=month_back)).strftime('%Y-%m'))
        test_window_ym.append((d - dateutil.relativedelta.relativedelta(months=month_back)).strftime('%Y-%m'))
    # aggregation 함수 선언
    agg_func = ['max', 'min', 'sum', 'mean', 'count', 'std', 'skew']
    # group by aggregation with Dictionary
    agg_dict = {
        'quantity': agg_func,
        'price': agg_func,
        'total': agg_func,
    }
    # general statistics for train data with time series trend
    for i, tr_ym in enumerate(train_window_ym):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['year_month'] >= tr_ym].groupby(['customer_id']).agg(
            agg_dict)  # 해당 year_month 이후부터 모든 데이터에 대한 aggregation을 실시
        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for level1, level2 in train_agg.columns:
            new_cols.append(f'{level1}-{level2}-{i}')
        train_agg.columns = new_cols
        train_agg.reset_index(inplace=True)

        if i == 0:
            train_data = train_agg
        else:
            train_data = train_data.merge(train_agg, on=['customer_id'], how='right')
    # general statistics for test data with time series trend
    for i, tr_ym in enumerate(test_window_ym):
        # group by aggretation 함수로 test 데이터 피처 생성
        test_agg = test.loc[test['year_month'] >= tr_ym].groupby(['customer_id']).agg(agg_dict)
        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for level1, level2 in test_agg.columns:
            new_cols.append(f'{level1}-{level2}-{i}')
        test_agg.columns = new_cols
        test_agg.reset_index(inplace=True)

        if i == 0:
            test_data = test_agg
        else:
            test_data = test_data.merge(test_agg, on=['customer_id'], how='right')
    return train_data, test_data


def add_seasonality(df, year_month):
    df = df.copy()
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    # train과 test 데이터 생성
    train = df[df['order_date'] < prev_ym]  # 2009-12부터 2011-10 데이터 추출
    test = df[df['order_date'] < year_month]  # 2009-12부터 2011-11 데이터 추출
    train_window_ym = []
    test_window_ym = []
    for month_back in [1, 6, 12, 18]:  # 각 주기성을 파악하고 싶은 구간을 생성
        train_window_ym.append(
            (
                (prev_ym - dateutil.relativedelta.relativedelta(months=month_back)).strftime('%Y-%m'),
                (prev_ym - dateutil.relativedelta.relativedelta(months=month_back + 2)).strftime('%Y-%m')
            # 1~3, 6~8, 12~14, 18~20 Pair를 만들어준다
            )
        )
        test_window_ym.append(
            (
                (d - dateutil.relativedelta.relativedelta(months=month_back)).strftime('%Y-%m'),
                (d - dateutil.relativedelta.relativedelta(months=month_back + 2)).strftime('%Y-%m')
            )
        )

    # aggregation 함수 선언
    agg_func = ['max', 'min', 'sum', 'mean', 'count', 'std', 'skew']
    # group by aggregation with Dictionary
    agg_dict = {
        'quantity': agg_func,
        'price': agg_func,
        'total': agg_func,
    }
    # seasonality for train data with time series
    for i, (tr_ym, tr_ym_3) in enumerate(train_window_ym):
        # group by aggretation 함수로 train 데이터 피처 생성
        # 구간 사이에 존재하는 월들에 대해서 aggregation을 진행
        train_agg = train.loc[(train['year_month'] >= tr_ym_3) & (train['year_month'] <= tr_ym)].groupby(
            ['customer_id']).agg(agg_dict)
        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for level1, level2 in train_agg.columns:
            new_cols.append(f'{level1}-{level2}-season{i}')
        train_agg.columns = new_cols
        train_agg.reset_index(inplace=True)

        if i == 0:
            train_data = train_agg
        else:
            train_data = train_data.merge(train_agg, on=['customer_id'], how='right')
    # seasonality for test data with time series
    for i, (tr_ym, tr_ym_3) in enumerate(test_window_ym):
        # group by aggretation 함수로 train 데이터 피처 생성
        test_agg = test.loc[(test['year_month'] >= tr_ym_3) & (test['year_month'] <= tr_ym)].groupby(
            ['customer_id']).agg(agg_dict)
        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for level1, level2 in test_agg.columns:
            new_cols.append(f'{level1}-{level2}-season{i}')
        test_agg.columns = new_cols
        test_agg.reset_index(inplace=True)

        if i == 0:
            test_data = test_agg
        else:
            test_data = test_data.merge(test_agg, on=['customer_id'], how='right')

    return train_data, test_data






def feature_engineering2(df, year_month):
    df = df.copy()

    # customer_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_cust_id'] = df.groupby(['customer_id'])['total'].cumsum()
    df['cumsum_quantity_by_cust_id'] = df.groupby(['customer_id'])['quantity'].cumsum()
    df['cumsum_price_by_cust_id'] = df.groupby(['customer_id'])['price'].cumsum()

    # product_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_prod_id'] = df.groupby(['product_id'])['total'].cumsum()
    df['cumsum_quantity_by_prod_id'] = df.groupby(['product_id'])['quantity'].cumsum()
    df['cumsum_price_by_prod_id'] = df.groupby(['product_id'])['price'].cumsum()

    # order_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_order_id'] = df.groupby(['order_id'])['total'].cumsum()
    df['cumsum_quantity_by_order_id'] = df.groupby(['order_id'])['quantity'].cumsum()
    df['cumsum_price_by_order_id'] = df.groupby(['order_id'])['price'].cumsum()

    # oredr_ts
    df['order_ts'] = df['order_date'].astype(np.int64)//1e9
    df['order_ts_diff'] = df.groupby(['customer_id'])['order_ts'].diff()
    df['quantity_diff'] = df.groupby(['customer_id'])['quantity'].diff()
    df['price_diff'] = df.groupby(['customer_id'])['price'].diff()
    df['total_diff'] = df.groupby(['customer_id'])['total'].diff()

    # mode
    df['month-mode'] = df['order_date'].dt.month
    df['year_month-mode'] = df['order_date'].dt.strftime('%Y-%m')

    # oredr_ts_plus ===
    df['order_ts_plus'] = df[df['total'] > 0]['order_date'].astype(np.int64) // 1e9
    df['order_ts_plus_diff'] = df[df['total'] > 0].groupby(['customer_id'])['order_ts'].diff()
    df['order_ts_plus'] = df['order_ts_plus'].fillna(0)
    df['order_ts_plus_diff'] = df['order_ts_plus_diff'].fillna(0)
    # df[~(df.order_id.str.contains('C'))].groupby(['customer_id'])['order_date'].last().astype(np.int64) // 1e9

    # ================================================================================================
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')

    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]

    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id', 'year_month', 'label']]
    test_label = generate_label(df, year_month)[['customer_id', 'year_month', 'label']]




    # ================================================================================================
    # 연월 피처 생성
    target = datetime.datetime.strptime('2011-11', "%Y-%m")  # 타겟 연월
    prev = target - dateutil.relativedelta.relativedelta(years=1)  # 전년 연월
    prev = prev.strftime('%Y-%m')  # 문자열로 변환
    groupby = train.groupby(['customer_id', 'year_month-mode'])['total'].sum()  # 고객별, 월별 total 합
    groupby = groupby.unstack()  # 월별을 컬럼으로 변환
    prev_pprev_total = groupby.loc[:, [prev]]  # 전년, 전전년 데이터만 추출
    prev_pprev_total = prev_pprev_total.fillna(0)

    train_1224 = (prev_pprev_total['2010-11']) / 2


    target = datetime.datetime.strptime('2011-12', "%Y-%m")  # 타겟 연월
    prev = target - dateutil.relativedelta.relativedelta(years=1)  # 전년 연월
    pprev = prev - dateutil.relativedelta.relativedelta(years=1)  # 전전년 연월
    prev, pprev = prev.strftime('%Y-%m'), pprev.strftime('%Y-%m')  # 문자열로 변환
    groupby = test.groupby(['customer_id', 'year_month-mode'])['total'].sum()  # 고객별, 월별 total 합
    groupby = groupby.unstack()  # 월별을 컬럼으로 변환
    prev_pprev_total = groupby.loc[:, [prev, pprev]]  # 전년, 전전년 데이터만 추출
    prev_pprev_total = prev_pprev_total.fillna(0)

    test_1224 = (prev_pprev_total['2010-12'] + prev_pprev_total['2009-12']) / 2


    # ================================================================================================

    # lambda 식
    mode_f = lambda x: x.value_counts().index[0]

    # group by aggregation 함수 선언
    agg_func = ['mean', 'max', 'min', 'sum', 'count', 'std', 'skew']
    # agg_func = ['mean', 'max'] # , 'min', 'sum', 'count', 'std', 'skew']
    agg_dict = {
        'order_ts': ['first', 'last'],
        'order_ts_diff': agg_func,
        'order_ts_plus': ['first', 'last'],
        'order_ts_plus_diff': agg_func,
        'quantity_diff': agg_func,
        'price_diff': agg_func,
        'total_diff': agg_func,
        'quantity': agg_func,
        'price': agg_func,
        'total': agg_func,
        'cumsum_total_by_cust_id': agg_func,
        'cumsum_quantity_by_cust_id': agg_func,
        'cumsum_price_by_cust_id': agg_func,
        'cumsum_total_by_prod_id': agg_func,
        'cumsum_quantity_by_prod_id': agg_func,
        'cumsum_price_by_prod_id': agg_func,
        'cumsum_total_by_order_id': agg_func,
        'cumsum_quantity_by_order_id': agg_func,
        'cumsum_price_by_order_id': agg_func,
        'order_id': ['nunique'],
        'product_id': ['nunique'],
        'month-mode': [mode_f],
        'year_month-mode': [mode_f],
    }
    all_train_data = pd.DataFrame()

    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_dict)

        new_cols = []
        for col in agg_dict.keys():
            for stat in agg_dict[col]:
                if type(stat) is str:
                    new_cols.append(f'{col}-{stat}')
                else:
                    new_cols.append(f'{col}-mode')
        train_agg.columns = new_cols
        train_agg.reset_index(inplace=True)

        train_agg['year_month'] = tr_ym

        all_train_data = all_train_data.append(train_agg)

    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    all_train_data['cycle_1224'] = train_1224.to_numpy()

    # ================================================================================================

    data = pd.read_csv("/opt/ml/code/input/train.csv", parse_dates=["order_date"])
    # # baseline feature engineering
    # train, test, y, features = feature_engineering(data, '2011-12')
    # trend
    train_t, test_t = add_trend(data, year_month='2011-12')
    # seasonality
    train_s, test_s = add_seasonality(data, year_month='2011-12')
    # train 데이터 병합
    all_train_data = all_train_data.merge(train_t, on=['customer_id'], how='left')
    all_train_data = all_train_data.merge(train_s, on=['customer_id'], how='left')
    all_train_data = all_train_data.fillna(0)

    # ================================================================================================

    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    print(features.shape)

    import csv
    with open("../output/feature.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for items in features.tolist():
            print(items)
            writer.writerow([items])

    test_agg = test.groupby(['customer_id']).agg(agg_dict)
    test_agg.columns = new_cols
    test_agg['cycle_1224'] = test_1224

    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')

    # test 데이터 병합 ===================================================================================
    test_data = test_data.merge(test_t, on=['customer_id'], how='left')
    test_data = test_data.merge(test_s, on=['customer_id'], how='left')
    test_data = test_data.fillna(0)

    # train, test 데이터 전처리
    print(all_train_data.shape)
    print(test_data.shape)
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)

    return x_tr, x_te, all_train_data['label'], features

def feature_engineering3(df, year_month):

    my_pick = [
        'order_ts-last',
        'order_ts-first',
        'price_diff-skew',
        'price-skew',
        'order_ts_diff-max',
        'quantity_diff-skew',
        'cumsum_total_by_prod_id-skew',
        'cumsum_price_by_prod_id-skew',
        'cumsum_total_by_cust_id-skew',
        'cumsum_quantity_by_prod_id-sum',
        'quantity-skew',
        'cumsum_total_by_order_id-skew',
        'cumsum_price_by_cust_id-skew',
        'cumsum_price_by_order_id-skew',
        'year_month-mode',
        'total_diff-skew',
        'price-mean',
        'cumsum_quantity_by_order_id-skew',
        'cumsum_quantity_by_prod_id-skew',
        'price_diff-mean',
    ]

    df = df.copy()

    # customer_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_cust_id'] = df.groupby(['customer_id'])['total'].cumsum()
    df['cumsum_quantity_by_cust_id'] = df.groupby(['customer_id'])['quantity'].cumsum()
    df['cumsum_price_by_cust_id'] = df.groupby(['customer_id'])['price'].cumsum()

    # product_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_prod_id'] = df.groupby(['product_id'])['total'].cumsum()
    df['cumsum_quantity_by_prod_id'] = df.groupby(['product_id'])['quantity'].cumsum()
    df['cumsum_price_by_prod_id'] = df.groupby(['product_id'])['price'].cumsum()

    # order_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_order_id'] = df.groupby(['order_id'])['total'].cumsum()
    df['cumsum_quantity_by_order_id'] = df.groupby(['order_id'])['quantity'].cumsum()
    df['cumsum_price_by_order_id'] = df.groupby(['order_id'])['price'].cumsum()

    # oredr_ts
    df['order_ts'] = df['order_date'].astype(np.int64)//1e9
    df['order_ts_diff'] = df.groupby(['customer_id'])['order_ts'].diff()
    df['quantity_diff'] = df.groupby(['customer_id'])['quantity'].diff()
    df['price_diff'] = df.groupby(['customer_id'])['price'].diff()
    df['total_diff'] = df.groupby(['customer_id'])['total'].diff()

    # mode
    df['month-mode'] = df['order_date'].dt.month
    df['year_month-mode'] = df['order_date'].dt.strftime('%Y-%m')

    # oredr_ts_plus ===
    df['order_ts_plus'] = df[df['total'] > 0]['order_date'].astype(np.int64) // 1e9
    df['order_ts_plus_diff'] = df[df['total'] > 0].groupby(['customer_id'])['order_ts'].diff()
    df['order_ts_plus'] = df['order_ts_plus'].fillna(0)
    df['order_ts_plus_diff'] = df['order_ts_plus_diff'].fillna(0)
    # df[~(df.order_id.str.contains('C'))].groupby(['customer_id'])['order_date'].last().astype(np.int64) // 1e9

    # ================================================================================================
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')

    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]

    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id', 'year_month', 'label']]
    test_label = generate_label(df, year_month)[['customer_id', 'year_month', 'label']]

    ####################################################################################

    # year_month 이전 월 계산
    bef_12_d1 = datetime.datetime.strptime(year_month, "%Y-%m")
    bef_12_prev_ym1 = bef_12_d1 - dateutil.relativedelta.relativedelta(months=12)
    bef_12_prev_ym1 = bef_12_prev_ym1.strftime('%Y-%m')
    merge_df_12_train = make_time_series_data(train, train, bef_12_prev_ym1, "%Y-%m")
    print(bef_12_prev_ym1)

    bef_24_d1 = datetime.datetime.strptime(year_month, "%Y-%m")
    bef_24_prev_ym1 = bef_24_d1 - dateutil.relativedelta.relativedelta(months=24)
    bef_24_prev_ym1 = bef_24_prev_ym1.strftime('%Y-%m')
    merge_df_24_train = make_time_series_data(train, train, bef_24_prev_ym1, "%Y-%m")
    print(bef_24_prev_ym1)

    merge_1224_train = merge_df_24_train.merge(merge_df_12_train, on=['customer_id'], how='left')
    series_1224_train = (merge_1224_train['total_x'] + merge_1224_train['total_y']) / 2

    ####################################################################################
    # year_month 이전 월 계산
    bef_12_d2 = datetime.datetime.strptime(prev_ym, "%Y-%m")
    bef_12_prev_ym2 = bef_12_d2 - dateutil.relativedelta.relativedelta(months=12)
    bef_12_prev_ym2 = bef_12_prev_ym2.strftime('%Y-%m')
    merge_df_12_test = make_time_series_data(test, test, bef_12_prev_ym2, "%Y-%m")
    print(bef_12_prev_ym2)

    bef_24_d2 = datetime.datetime.strptime(prev_ym, "%Y-%m")
    bef_24_prev_ym2 = bef_24_d2 - dateutil.relativedelta.relativedelta(months=24)
    bef_24_prev_ym2 = bef_24_prev_ym2.strftime('%Y-%m')
    merge_df_24_test = make_time_series_data(test, test, bef_24_prev_ym2, "%Y-%m")
    print(bef_24_prev_ym2)

    merge_1224_test = merge_df_24_test.merge(merge_df_12_test, on=['customer_id'], how='left')
    series_1224_test = (merge_1224_test['total_x'] + merge_1224_test['total_y']) / 2

    ####################################################################################

    # lambda 식
    mode_f = lambda x: x.value_counts().index[0]

    # group by aggregation 함수 선언
    # agg_func = ['mean', 'max', 'min', 'sum', 'count', 'std', 'skew']
    agg_func = ['mean', 'max'] # , 'min', 'sum', 'count', 'std', 'skew']
    agg_dict = {
        'order_ts': ['first', 'last'],
        'order_ts_diff': agg_func,
        # 'order_ts_plus': ['first', 'last'],
        # 'order_ts_plus_diff': agg_func,
        # 'quantity_diff': agg_func,
        # 'price_diff': agg_func,
        # 'total_diff': agg_func,
        # 'quantity': agg_func,
        # 'price': agg_func,
        # 'total': agg_func,
        # 'cumsum_total_by_cust_id': agg_func,
        # 'cumsum_quantity_by_cust_id': agg_func,
        # 'cumsum_price_by_cust_id': agg_func,
        # 'cumsum_total_by_prod_id': agg_func,
        # 'cumsum_quantity_by_prod_id': agg_func,
        # 'cumsum_price_by_prod_id': agg_func,
        # 'cumsum_total_by_order_id': agg_func,
        # 'cumsum_quantity_by_order_id': agg_func,
        # 'cumsum_price_by_order_id': agg_func,
        # 'order_id': ['nunique'],
        # 'product_id': ['nunique'],
        # 'month-mode': [mode_f],
        # 'year_month-mode': [mode_f],
    }
    all_train_data = pd.DataFrame()

    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_dict)

        new_cols = []
        for col in agg_dict.keys():
            for stat in agg_dict[col]:
                if type(stat) is str:
                    new_cols.append(f'{col}-{stat}')
                else:
                    new_cols.append(f'{col}-mode')
        train_agg.columns = new_cols
        train_agg.reset_index(inplace=True)

        train_agg['year_month'] = tr_ym

        all_train_data = all_train_data.append(train_agg)

    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    all_train_data['cycle_1224'] = series_1224_train
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns

    import csv
    with open("../output/feature.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for items in features.tolist():
            print(items)
            writer.writerow([items])

    test_agg = test.groupby(['customer_id']).agg(agg_dict)
    test_agg.columns = new_cols
    test_agg['cycle_1224'] = series_1224_test

    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')

    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    # x_tr = x_tr[my_pick]
    # x_te = x_te[my_pick]
    # features = pd.Index(my_pick)
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)

    return x_tr, x_te, all_train_data['label'], features



if __name__ == '__main__':
    
    print('data_dir', data_dir)
