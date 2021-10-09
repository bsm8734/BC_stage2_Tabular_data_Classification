# library import

# Data manipulation
import pandas as pd 
import numpy as np
import datetime
import dateutil.relativedelta

# train data load

train_data_path = "/opt/ml/code/input/train.csv"
train_df = pd.read_csv(train_data_path, parse_dates=["order_date"])

"""
def extract_char(str_list : set) -> set

'''문자 + 숫자 에 어떤 문자가 있는 지 패턴 파악'''
"""

#문자열 리스트에서 문자+숫자인 변수에서 어떤 문자가 있는 지 추출
def extract_char(str_list : set) -> set:
    '''
        :str_list: set  (숫자/ 숫자+문자 가 섞인 문자열(주로 id) 집합)
        :return: set    (숫자+문자 문자열에서 문자열 추출)
    '''
    char_set = set([])
    for word in str_list:
        str = ""
        for idx in range(len(word)) :
            if not word[idx].isdigit() :
                str = str + word[idx]
            else:
                str = str + "_"
        if str not in char_set :
            char_set.add(str)
    return char_set




# def print_element(myset:set, term=5) :: 일정 개수만큼 잘라서 출력

# 한 줄에 출력할 개수 = 5개로 지정
def print_element(myset : set, term=5):
    for idx, pid in enumerate(myset):
        if ((idx+1) % term == 0) :
            print(pid)
        else :
            print(pid, end = "\t")





# order_id 조사

# order_id에서 문자가 포함된 id 추출
order_id_list = train_df['order_id'].tolist()
order_id_char_list = [id for id in order_id_list if not id.isdigit()]
# 문자 포함된 id 에 어떤 글자가 포함되어 있는 지 확인
order_id_char_set = set(order_id_char_list)
char_set = extract_char(order_id_char_set)  #함수 --> id에 문자가 포함된 경우 패턴 파악

print("number of order_id included char =", len(order_id_char_list))
print("number of order_id included char(no duplicated) =", len(order_id_char_set))
#문자확인 -> C 밖에 없다.
print("char_set = ", char_set)





# product_id 조사

# product_id에서 문자가 포함된 id 추출
pid_list = train_df['product_id'].tolist()
pid_char_list = [id for id in pid_list if not id.isdigit()]

# 문자 포함된 id 에 어떤 글자가 포함되어 있는 지 확인
pid_char_set = set(pid_char_list)
pid_char_patterns = extract_char(pid_char_set)  #함수 --> id에 문자가 포함된 경우 패턴 파악

print("number of pid included char =", len(pid_char_list))
print("number of pid included char(no duplicated) =", len(pid_char_set))
print("len(pid_patterns) :", len(pid_char_patterns))
print()


#문자확인(5개씩 끊어서 출력)
print_element(pid_char_patterns)

