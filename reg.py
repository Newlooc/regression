import csv
import time
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import numpy as np

keys = ('AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR')

#Gather Value of PM2.5 every hour
def getValue(comp) :
    value = []
    f = open("data/train.csv")
    readers = csv.reader(f)
    for itemKey, itemValue in enumerate(readers) :
        if itemKey > 0 and itemValue[1] == comp :
            for columnKey, columnValue in enumerate(itemValue) :
                if columnKey > 1 :
                    try :
                        value.append(float(columnValue))
                    except ValueError :
                        value.append(float(0))
    return value

pm25 = getValue('PM2.5')
pm10 = getValue('PM10')
rain = getValue('RAINFALL')
so2  = getValue('SO2')

#前6推7
rangeOfDataKey = range(len(pm25) - 6)
w1 = w2 = w3 = w4 = w5 = w6 = b =1
w1gpart1 = w2gpart1 = w3gpart1 = w4gpart1 = w5gpart1 = w6gpart1 = bgpart1 = 0
w1gpart2 = w2gpart2 = w3gpart2 = w4gpart2 = w5gpart2 = w6gpart2 = bgpart2 = 0
w1g = w2g = w3g = w4g = w5g = w6g = bg = 0
adw1 = adw2 = adw3 = adw4 = adw5 = adw6 = adb = 0
loss = 0
loss_h = []
lr = 1

for n in range(100) :
    #loss function中的求和
    for i in rangeOfDataKey :
        #求梯度
        part1 = 2 * (pm25[i + 6] - (w1 * pm25[i] + w2 * pm25[i + 1] + w3 * pm25[i + 2] + w4 * pm25[i + 3] + w5 * pm25[i + 4] + w6 * pm25[i + 5] + b))
        #w1
        w1gpart2 = -1 * pm25[i]
        w1g += part1 * w1gpart2
        #w2
        w2gpart2 = -1 * pm25[i+1]
        w2g += part1 * w2gpart2
        #w3
        w3gpart2 = -1 * pm25[i+2]
        w3g += part1 * w3gpart2
        #w4
        w4gpart2 = -1 * pm25[i+3]
        w4g += part1 * w4gpart2
        #w5
        w5gpart2 = -1 * pm25[i+4]
        w5g += part1 * w5gpart2
        #w6
        w6gpart2 = -1 * pm25[i+5]
        w6g += part1 * w6gpart2
        #b
        bgpart2 = -1
        bg += part1 * bgpart2
        
        loss += (part1 / 2)**2

    loss_h.append(loss)
    loss = 0
    #adagrad 根号下部分
    adw1 += w1g**2
    adw2 += w2g**2
    adw3 += w3g**2
    adw4 += w4g**2
    adw5 += w5g**2
    adw6 += w6g**2
    adb += bg**2

    w1 -= lr / np.sqrt(adw1) * w1g
    w2 -= lr / np.sqrt(adw2) * w2g
    w3 -= lr / np.sqrt(adw3) * w3g
    w4 -= lr / np.sqrt(adw4) * w4g
    w5 -= lr / np.sqrt(adw5) * w5g
    w6 -= lr / np.sqrt(adw6) * w6g
    b -= lr / np.sqrt(adb) * bg

print(w1, w2, w3, w4, w5, w6, b)
#exit


res=[]
for i in rangeOfDataKey : 
    res.append(w1 * pm25[i] + w2 * pm25[i + 1] + w3 * pm25[i + 2] + w4 * pm25[i + 3] + w5 * pm25[i + 4] + w6 * pm25[i + 5] + b)


x = range(len(loss_h))
#plt.plot(x, loss_h, label='loss')
plt.plot(range(200), res[0:200], label='res')
plt.plot(range(200), pm25[0:200], label='PM2.5')
#plt.plot(x, pm10[0:200], label='PM10')
#plt.plot(x, rain[0:200], label='Rain')
#plt.plot(x, so2[0:200], label='SO2')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple Plot")

plt.legend()

plt.show()




