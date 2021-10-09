import pandas as pd
from sklearn.metrics import roc_auc_score
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np

# 평가 함수
def evaluation(gt_path, pred_path):
    # Ground Truth 경로에서 정답 파일 읽기
    label = pd.read_csv(gt_path + '/label.csv')['label']
    
    # 테스트 결과 예측 파일 읽기
    preds = pd.read_csv(pred_path + '/output.csv')['probability']

    # AUC 스코어 계산
    score = roc_auc_score(label, preds)


    
    return score

#평가 지표
def metrics(gt_path, pred_path):
    # Ground Truth 경로에서 정답 파일 읽기
    y_test = pd.read_csv(gt_path + '/label.csv')['label']

    # 테스트 결과 예측 파일 읽기
    pred = pd.read_csv(pred_path + '/output.csv')['probability']

    accuracy = metrics.accuracy_score(y_test,pred)
    precision = metrics.precision_score(y_test,pred)
    recall = metrics.recall_score(y_test,pred)
    f1 = metrics.f1_score(y_test,pred)
    roc_score = metrics.roc_auc_score(y_test,pred,average='macro')

    print('정확도 : {0:.2f}, 정밀도 : {1:.2f}, 재현율 : {2:.2f}'.format(accuracy,precision,recall))
    print('f1-score : {0:.2f}, auc : {1:.2f}'.format(f1,roc_score,recall))


def plot_feature_importances(df, n=20, color='blue', figsize=(12, 8)):
    # 피처 중요도 순으로 내림차순 정렬
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)

    # 피처 중요도 정규화 및 누적 중요도 계산
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    plt.rcParams['font.size'] = 12
    plt.style.use('fivethirtyeight')
    # 피처 중요도 순으로 n개까지 바플롯으로 그리기
    df.loc[:n, :].plot.barh(y='importance_normalized',
                            x='feature', color=color,
                            edgecolor='k', figsize=figsize,
                            legend=False)

    plt.xlabel('Normalized Importance', size=18)
    plt.ylabel('')
    plt.title(f'Top {n} Most Important Features', size=18)
    plt.gca().invert_yaxis()

    return df


if __name__ == '__main__':
    gt_path = '../input'
    pred_path = '.'
    
    print(evaluation(gt_path, pred_path))
