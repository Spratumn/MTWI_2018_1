#!/usr/bin/python
#coding:utf-8 
"""
@author: SPATU
@software: PyCharm
@file: get_train_data.py
@time: 2019/1/4 9:18
"""
import cv2 as cv
import math
import numpy as np
import pickle


# 自定义点类，用于处理矩形框的四个点信息
class Point():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


# 通过坐标信息计算矩形边长：高和宽
def get_edge_len(local):
    h1 = math.sqrt(math.pow((local[0]-local[2]), 2) + math.pow((local[1]-local[3]), 2))
    h2 = math.sqrt(math.pow((local[4] - local[6]), 2) + math.pow((local[5] - local[7]), 2))

    w1 = math.sqrt(math.pow((local[0] - local[6]), 2) + math.pow((local[1] - local[7]), 2))
    w2 = math.sqrt(math.pow((local[2] - local[4]), 2) + math.pow((local[3] - local[5]), 2))

    height = max([h1, h2])
    width = max([w1, w2])
    return int(height), int(width)


complete = 0
label_list = []
ori_img = np.zeros([100, 100, 3])
temp = {'image': ori_img, 'label': 'empty'}
train_data1 = []
train_data2 = []
train_data3 = []
train_data4 = []
# 获取的数据数量
index = 0
# 处理number张图片
number = 1000
split_num = number/100
for num in range(number):
    # 事先使用了电脑的批量重命名方法将数据集文件重命名为比较容易迭代的方式
    filename = 'txt_train/train_txt ('+str(num+1)+').txt'
    image_name = 'image_train/train_image ('+str(num+1)+').jpg'
    img = cv.imread(image_name)
    with open(filename, 'r', encoding='UTF-8') as f:
        lst = (f.readlines())
    for label in lst:
        info = label.split(',')
        if info[8].replace("\n", "") == '###':
            pass
        else:
            loca = []
            for i in range(8):
                loca.append(int(float(info[i])))
            p1 = Point(loca[0], loca[1])
            p2 = Point(loca[2], loca[3])
            p3 = Point(loca[4], loca[5])
            p4 = Point(loca[6], loca[7])
            cx = int(sum([loca[0], loca[2], loca[4], loca[6]])/4)
            cy = int(sum([loca[1], loca[3], loca[5], loca[7]])/4)
            p_center = Point(cx, cy)
            # label给出的点坐标值并不是标准的顺序，需要排定顺序
            top = []  # 上部两个点
            bottom = []  # 下部两个点
            try:
                for p in [p1, p2, p3, p4]:
                    if p.x < p_center.x:
                        top.append(p)
                    else:
                        bottom.append(p)
                if top[0].y > top[1].y:  # 判断左上角和右上角
                    top[0], top[1] = top[1], top[0]
                if bottom[0].y > bottom[1].y:  # 判断左下角和右下角
                    bottom[0], bottom[1] = bottom[1], bottom[0]
            except IndexError:
                # print('坐标错误')，信息错误无法形成矩形框
                break
            point_value = [top[0].x, top[0].y,
                          bottom[0].x, bottom[0].y,
                          bottom[1].x, bottom[1].y,
                          top[1].x, top[1].y]
            h, w = get_edge_len(point_value)

            # 获取矩形图片
            pts1 = np.float32([[top[0].x, top[0].y], [bottom[0].x, bottom[0].y], [top[1].x, top[1].y]])
            pts2 = np.float32([[0, 0], [h, 0], [0, w]])
            M = cv.getAffineTransform(pts1, pts2)
            try:
                dst = cv.warpAffine(img, M, (h, w))
            except BaseException:
                break
            # 保存图片，保存到单独的图片文件夹
            # cv.imwrite('train_data/'+str(index)+'.jpg', dst)
            # label_list.append(info[8])
            # 分成四个文件，确保每个文件数据量比较接近
            if num <= 2560:
                train_data1.append({'image': dst, 'label': info[8]})
                index += 1
            if 2560 < num <= 5012:
                train_data2.append({'image': dst, 'label': info[8]})
                index += 1
            if 5012 < num <= 7510:
                train_data3.append({'image': dst, 'label': info[8]})
                index += 1
            if 7510 < num:
                train_data4.append({'image': dst, 'label': info[8]})
                index += 1
    if num % split_num == 0:
        print('complete ' + str(complete)+'%')
        complete += 1
# 保存label信息
# with open('train_data/'+'label.txt', 'w', encoding='UTF-8') as f:
#     for label in label_list:
#         f.write(label)
if complete == 100:
    print('complete ' + str(complete) + '%')
pickle_file = open('train_data_patch1.pkl', 'wb')
print('Saving the first patch to file...')
pickle.dump(train_data1, pickle_file)
pickle_file.close()
pickle_file = open('train_data_patch2.pkl', 'wb')
print('Saving the second patch to file...')
pickle.dump(train_data2, pickle_file)
pickle_file.close()
pickle_file = open('train_data_patch3.pkl', 'wb')
print('Saving the third patch to file...')
pickle.dump(train_data3, pickle_file)
pickle_file.close()
pickle_file = open('train_data_patch4.pkl', 'wb')
print('Saving the forth patch to file...')
pickle.dump(train_data4, pickle_file)
pickle_file.close()
print('Done!')


if __name__ == '__main__':
    # 测试数据
    # 读取数据
    pickle_file = open('train_data_patch1.pkl', 'rb')
    train_data1 = pickle.load(pickle_file)
    pickle_file.close()

    pickle_file = open('train_data_patch2.pkl', 'rb')
    train_data2 = pickle.load(pickle_file)
    pickle_file.close()

    pickle_file = open('train_data_patch3.pkl', 'rb')
    train_data3 = pickle.load(pickle_file)
    pickle_file.close()

    pickle_file = open('train_data_patch4.pkl', 'rb')
    train_data4 = pickle.load(pickle_file)
    pickle_file.close()

    print(len(train_data1))
    print(len(train_data2))
    print(len(train_data3))
    print(len(train_data4))
