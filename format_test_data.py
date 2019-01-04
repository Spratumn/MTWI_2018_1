#!/usr/bin/python
#coding:utf-8 
"""
@author: SPATU
@software: PyCharm
@file: format_test_data.py
@time: 2019/1/4 9:35
"""
import pickle
import cv2 as cv
import numpy as np
# 将测试集转换为5个数据集
test_data1 = []
test_data2 = []
test_data3 = []
test_data4 = []
test_data5 = []
img = np.array([100, 100, 3], dtype=np.uint8)
for i in range(148734):
    img = cv.imread('test_data/test_data/line_' + str(i) + '.jpg')
    if i <= 30000:
        test_data1.append(img)
    if 30000 < i <= 60000:
        test_data2.append(img)
    if 60000 < i <= 90000:
        test_data3.append(img)
    if 90000 < i <= 120000:
        test_data4.append(img)
    if 120000 < i:
        test_data5.append(img)

pickle_file = open('test_data_patch1.pkl', 'wb')
print('Saving the first patch to file...')
pickle.dump(test_data1, pickle_file)
pickle_file.close()

pickle_file = open('test_data_patch2.pkl', 'wb')
print('Saving the second patch to file...')
pickle.dump(test_data2, pickle_file)
pickle_file.close()

pickle_file = open('test_data_patch3.pkl', 'wb')
print('Saving the third patch to file...')
pickle.dump(test_data3, pickle_file)
pickle_file.close()

pickle_file = open('test_data_patch4.pkl', 'wb')
print('Saving the forth patch to file...')
pickle.dump(test_data4, pickle_file)
pickle_file.close()

pickle_file = open('test_data_patch5.pkl', 'wb')
print('Saving the fifth patch to file...')
pickle.dump(test_data5, pickle_file)
pickle_file.close()

print('Done!')
