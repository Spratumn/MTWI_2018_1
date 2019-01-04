#!/usr/bin/python
#coding:utf-8 
"""
@author: SpAtu
@software: PyCharm
@file: pickle_test.py
@time: 2018/12/23 15:24
"""
import pickle
import cv2 as cv


pickle_file = open('test_data/test_data_patch1.pkl', 'rb')
test_data1 = pickle.load(pickle_file)
pickle_file.close()

pickle_file = open('train_data/train_data_patch1.pkl', 'rb')
train_data1 = pickle.load(pickle_file)
pickle_file.close()

print('data test:')
train_image = train_data1[100]['image']
train_label = train_data1[100]['label']
test_image = test_data1[100]
cv.imshow('train_image', train_image)
cv.imshow('test_image', test_image)
print('label: ' + train_label)
cv.waitKey(0)
