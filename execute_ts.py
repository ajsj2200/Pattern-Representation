#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from operator import itemgetter
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import ts_dist
import pyximport # pyximport.install()
import seaborn as sns

from ts_dist import dtw_dist as dtw_dist_py
from ts_dist import lcss_dist as lcss_dist_py
from ts_dist import edr_dist as edr_dist_py
from datetime import datetime


def execute_ts_als(data, data_search, al, delta = np.inf, epsilon = 0.5):


    data = data
    data_search = data_search
    data_cal = 0
    find = []
    exe_time = 0
    delta = delta
    epsilon = epsilon
    
    if al == 'edr' : # EDR
        
        start = time.time()  # 시작 시간 저장
        for i in range(0, int((len(data)+1))):
            if i+WINDOW_SIZE <= len(data):
                data_cal = data.iloc[i:i+WINDOW_SIZE,]
                edr = edr_dist_py(data_search.T, data_cal.T, epsilon = epsilon) # 낮출수록 민감해진다
                diction = {'data_cal':data_cal, 'edr':edr, 'start_index': i}
                find.append(diction)
        exe_time = time.time() - start    
        
        find_sorted = sorted(find, key=itemgetter('edr'))

        edr_list = []
        for i in range(len(find_sorted)):
            edr_list.append(find_sorted[i]['edr'])

        index_list = []
        for i in range(len(find_sorted)):
            index_list.append(find_sorted[i]['start_index'])

        edr_list = pd.Series(edr_list)
        index_list = pd.Series(index_list)

        listed=  pd.concat((edr_list, index_list), axis = 1)
        listed.columns = ['edr', 'index_list']
        listed = listed.sort_values(by = 'index_list').reset_index(drop = True)
        listed = listed.drop(columns = 'index_list')
        
        return [exe_time, listed]
        
    if al == 'lcss' : # LCSS
        
        start = time.time()  # 시작 시간 저장
        for i in range(0, int((len(data)+1))):
            if i+WINDOW_SIZE <= len(data):
                data_cal = data.iloc[i:i+WINDOW_SIZE,]
                edr = lcss_dist_py(data_search.T, data_cal.T,delta = delta, epsilon = epsilon)
                diction = {'data_cal':data_cal, 'edr':edr, 'start_index': i}
                find.append(diction)
        exe_time = time.time() - start   
        
        find_sorted = sorted(find, key=itemgetter('edr'))

        edr_list = []
        for i in range(len(find_sorted)):
            edr_list.append(find_sorted[i]['edr'])

        index_list = []
        for i in range(len(find_sorted)):
            index_list.append(find_sorted[i]['start_index'])

        edr_list = pd.Series(edr_list)
        index_list = pd.Series(index_list)

        listed=  pd.concat((edr_list, index_list), axis = 1)
        listed.columns = ['lcss', 'index_list']
        listed = listed.sort_values(by = 'index_list').reset_index(drop = True)
        listed = listed.drop(columns = 'index_list')
        
        return [exe_time, listed]
    
    if al == 'dtw' : # DTW
        
        import fastdtw.fastdtw
        
        start = time.time()  # 시작 시간 저장
        for i in range(0, int((len(data)+1))):
            if i+WINDOW_SIZE <= len(data):
                data_cal = data.iloc[i:i+WINDOW_SIZE,]
                edr = fastdtw.fastdtw(data_search, data_cal)[0]
                diction = {'data_cal':data_cal, 'edr':edr, 'start_index': i}
                find.append(diction)
        exe_time = time.time() - start   
        
        find_sorted = sorted(find, key=itemgetter('edr'))

        edr_list = []
        for i in range(len(find_sorted)):
            edr_list.append(find_sorted[i]['edr'])

        index_list = []
        for i in range(len(find_sorted)):
            index_list.append(find_sorted[i]['start_index'])

        edr_list = pd.Series(edr_list)
        index_list = pd.Series(index_list)

        listed=  pd.concat((edr_list, index_list), axis = 1)
        listed.columns = ['dtw', 'index_list']
        listed = listed.sort_values(by = 'index_list').reset_index(drop = True)
        listed = listed.drop(columns = 'index_list')
        
        return [exe_time, listed]    

