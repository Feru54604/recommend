# coding: utf-8
import math
import numpy as np
from matplotlib import pyplot
import random

USER_NUM = 943
ITEM_NUM = 1682
DATA_NUM = 80000 
print "DIM?"
DIM = int(raw_input())
print "iter?"
iternum = int(raw_input())

#欠損値は0として行列化
rate = np.zeros([USER_NUM,ITEM_NUM])
datalist = []

#testrateにテストデータの評価値を格納
testrate = np.zeros([USER_NUM,ITEM_NUM])
for line in open("ml-100k/u1.test", 'r'):
    items = line.split('\t')
    user = int(items[0])-1
    movie = int(items[1])-1
    testrate[user][movie] = int(items[2])

#ファイルを読み込み行列を作る
def learn(filename):
    for line in open(filename, 'r'):
        items = line.split('\t')
        user = int(items[0])-1
        movie = int(items[1])-1
        rate[user][movie] = int(items[2])
        datalist.append([user,movie,rate[user][movie]])

learn("ml-100k/u1.base")
#特徴ベクトルの初期値を全て1に設定
user_feature = np.ones([USER_NUM,DIM])
item_feature = np.ones([ITEM_NUM,DIM])

#for i in range(USER_NUM):
#    for j in range(DIM):
#        user_feature[i][j] = random.random()
#for i in range(ITEM_NUM):
#    for j in range(DIM):
#        item_feature[i][j] = random.random()


#誤差を求める
def square_error():
    error = 0
    count = 0
    for i,v in enumerate(testrate):
        for j,w in enumerate(v): #rate[i][j]と特徴ベクトルの内積を比較
            if w != 0:
                error += (w-inner_product(user_feature[i],item_feature[j]))**2
                count += 1
    return error

def inner_product(a,b):
    total = 0
    for i in range(DIM):
        total += a[i] * b[i]
    if total > 5:
        return 5
    if total < 1:
        return 1
    return total

def update_parameter():
    for i,dat in enumerate(datalist): #user,movie,value
        uid = dat[0]
        iid = dat[1]
        val = dat[2]
        user_feature[uid] -= ALF * (-(val - user_feature[uid].T.dot(item_feature[iid]))*item_feature[iid])
        item_feature[iid] -= ALF * (-(val - user_feature[uid].T.dot(item_feature[iid]))*user_feature[uid])
        #user_feature[uid] -= ALF * (-(val - user_feature[uid].T.dot(item_feature[iid]))*item_feature[iid] + user_feature[uid]*0.01)
        #item_feature[iid] -= ALF * (-(val - user_feature[uid].T.dot(item_feature[iid]))*user_feature[uid] + item_feature[iid]*0.01)
prev = 100000000
errorlist = []
for i in range(1,iternum+1):
    ALF = 0.001
    update_parameter()
    ave_error = square_error()/20000
    print i,ave_error
    errorlist.append(ave_error)
    prev = ave_error

pyplot.plot(errorlist)
pyplot.show()

#テストを行う
#rate = [[0 for i in range(ITEM_NUM)] for j in range(USER_NUM)]
#learn("ml-100k/u1.test")
#er = 0
#ct = 0
#print square_error()/20000
