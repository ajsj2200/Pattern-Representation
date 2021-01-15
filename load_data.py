#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from operator import itemgetter
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def load_data():
    data = pd.read_csv('data_new.csv')
    date = datetime.strptime(data['data_date'][0], '%Y-%m-%d %H:%M:%S.%f').strftime('%m/%d/%Y')
    date = pd.DataFrame(data['data_date'])
    data['date'] = date.applymap(str).applymap(lambda s: '{}-{}-{} {}:{}:{}'.format(
        s[0:4], s[5:7], s[8:10],s[11:13],s[14:16],s[17:19]))
    data.index = pd.to_datetime(data['date'])
    data = data.drop(columns = ['data_date', 'date'])
    data = data['2020-12-09 00:00:00':'2020-12-09 23:59:59'] # # 2020-12-09일자 데이터 슬라이싱
    data.columns = ['oil_temperature']
    
    scaler = MinMaxScaler()
    data_s = scaler.fit_transform(data)
    data_s = pd.DataFrame(data_s)
    data_s.columns = data.columns
    data_s.index = data.index
    return data_s

