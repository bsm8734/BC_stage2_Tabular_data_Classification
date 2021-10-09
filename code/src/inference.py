# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta
import argparse

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

# Custom library
import utils
from utils import seed_everything, print_score
from features import generate_label, feature_engineering, feature_engineering1, feature_engineering2

TOTAL_THRES = 300 # 구매액 임계값
SEED = 42 # 랜덤 시드
seed_everything(SEED) # 시드 고정


data_dir = '../input' # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model' # os.environ['SM_MODEL_DIR']
output_dir = '../output' # os.environ['SM_OUTPUT_DATA_DIR']

def make_lgb_oof_prediction(train, y, test, features, categorical_features='auto', model_params=None, folds=10):
    x_train = train[features]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    # ============================================================================== 기하
    # test_preds = np.ones(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    # ============================================================================== 기하
    # score = 1
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # LightGBM 데이터셋 선언
        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dvalid = lgb.Dataset(x_val, label=y_val)
        
        # LightGBM 모델 훈련
        clf = lgb.train(
            model_params,
            dtrain,
            valid_sets=[dtrain, dvalid], # Validation 성능을 측정할 수 있도록 설정
            categorical_feature=categorical_features,
            verbose_eval=200
        )

        # Validation 데이터 예측
        val_preds = clf.predict(x_val)
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 측정
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장  # 기하평군으로 바꾸면 잘된다.
        score += roc_auc_score(y_val, val_preds) / folds
        # ============================================================================== 기하
        # score *= roc_auc_score(y_val, val_preds)
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(x_test) / folds
        # ============================================================================== 기하
        # test_preds *= (clf.predict(x_test) + np.finfo(float).eps)
        
        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold+1}'] = clf.feature_importance()

        del x_tr, x_val, y_tr, y_val
        gc.collect()

    # ============================================================================== 기하
    # score = score**(1/folds)
    # test_preds = test_preds**(1/folds)

    print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
        
    # 폴드별 피처 중요도 평균값 계산해서 저장 
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi



if __name__ == '__main__':
    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])

    # 예측할 연월 설정
    year_month = '2011-12'

    # model_params = {
    #     'objective': 'binary',  # 이진 분류
    #     'boosting_type': 'gbdt',
    #     'metric': 'auc',  # 평가 지표 설정
    #     'feature_fraction': 0.7,  # 피처 샘플링 비율
    #     'bagging_fraction': 0.7,  # 데이터 샘플링 비율
    #     'bagging_freq': 1,
    #     'n_estimators': 10000,  # 트리 개수
    #     'early_stopping_rounds': 1400,
    #     'learning_rate': 0.01,
    #     'max_bin':255,
    #     'seed': SEED,
    #     'verbose': -1,
    #     'n_jobs': -1,
    #     'num_leaves': 31,
    #     'min_data_in_leaf':1500,
    #     'lambda_l1': 1,
    #     'lambda_l2':1,
    #     # 'boost_from_average': False,
    # }

    model_params = {
        'objective': 'binary',  # 이진 분류
        'boosting_type': 'gbdt',
        'metric': 'auc',  # 평가 지표 설정
        'feature_fraction': 0.8,  # 피처 샘플링 비율
        'bagging_fraction': 0.8,  # 데이터 샘플링 비율
        'bagging_freq': 1,
        'n_estimators': 10000,  # 트리 개수
        'early_stopping_rounds': 100,
        'seed': SEED,
        'verbose': -1,
        'n_jobs': -1,
        #
        # 'num_leaves': 64,
        # 'boost_from_average': False,
    }

    # 피처 엔지니어링 실행
    import features
    # features.get_year_month_list(data, year_month)
    # print('end')
    train, test, y, features = feature_engineering2(data, year_month) ##################

    # print(train.head())
    # print(test.head())
    # print(y.head())
    # print(features.shape)

    # SMOTE 실험
    # from imblearn.over_sampling import SMOTE
    #
    # t = train.drop(['customer_id', 'year_month', 'label'], axis=1)
    # X_train_over, y_train_over = SMOTE(random_state=0).fit_resample(t, y)
    # print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', t.shape, y.shape)
    # print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_over.shape, y_train_over.shape)
    # print('SMOTE 적용 후 레이블 값 분포:\n', pd.Series(y_train_over).value_counts())

    # Quantile 실험
    from sklearn.preprocessing import QuantileTransformer
    # quan = QuantileTransformer(n_quantiles=2000, output_distribution='normal',random_state=42)
    quan = QuantileTransformer(n_quantiles=1000, random_state=42)
    X_quan = quan.fit_transform(train[features])
    Y_quan = quan.fit_transform(test[features])
    x_quan = pd.DataFrame(X_quan, columns=features)
    y_quan = pd.DataFrame(Y_quan, columns=features)
    train[features] = x_quan[features]
    test[features] = y_quan[features]

    ### feature_pca = feature.append(train_pca_df.columns)

    # Cross Validation Out Of Fold로 LightGBM 모델 훈련 및 예측
    y_oof, test_preds, fi = make_lgb_oof_prediction(train, y, test, features, model_params=model_params)
    # y_oof, test_preds, fi = make_lgb_oof_prediction(X_train_over, y_train_over, test, features, model_params=model_params) # SMOTE
    # y_oof, test_preds, fi = make_lgb_oof_prediction(train, y, test, features, model_params=model_params) # QUANT

    sub = pd.read_csv(data_dir + '/sample_submission.csv')
    sub['probability'] = test_preds # 테스트 예측 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    sub.to_csv(os.path.join(output_dir , 'output17.csv'), index=False) # 제출 파일 쓰기  # 상황: 아직 없음 # 경로 바꿔주기!!!!

    from evaluation import plot_feature_importances
    fi = plot_feature_importances(fi)