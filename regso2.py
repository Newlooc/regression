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
#参数
w1 = w2 = w3 = w4 = w5 = w6 = b = 1
ws1 = ws2 = ws3 = ws4 = ws5 = ws6 = 1
#导数第一部分
part1 = 0
#Ada
adw1 = adw2 = adw3 = adw4 = adw5 = adw6 = adb = 0
adws1 = adws2 = adws3 = adws4 = adws5 = adws6 = 0

loss = 0
loss_h = []
lr = 1

for n in range(1000) :
    #导数第二部分
    w1gpart2 = w2gpart2 = w3gpart2 = w4gpart2 = w5gpart2 = w6gpart2 = bgpart2 = 0
    ws1gpart2 = ws2gpart2 = ws3gpart2 = ws4gpart2 = ws5gpart2 = ws6gpart2 = 0
    #梯度
    w1g = w2g = w3g = w4g = w5g = w6g = bg = 0
    ws1g = ws2g = ws3g = ws4g = ws5g = ws6g = 0
    #loss function中的求和
    for i in rangeOfDataKey :
        #求梯度
        part1 = 2 * (pm25[i + 6] - ( \
            w1 * pm25[i] + \
            w2 * pm25[i + 1] + \
            w3 * pm25[i + 2] + \
            w4 * pm25[i + 3] + \
            w5 * pm25[i + 4] + \
            w6 * pm25[i + 5] + \
            ws1 * pm10[i] + \
            ws2 * pm10[i + 1] + \
            ws3 * pm10[i + 2] + \
            ws4 * pm10[i + 3] + \
            ws5 * pm10[i + 4] + \
            ws6 * pm10[i + 5] + \
            b))
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
        #ws1
        ws1gpart2 = -1 * pm10[i]
        ws1g += part1 * ws1gpart2
        #ws2
        ws2gpart2 = -1 * pm10[i+1]
        ws2g += part1 * ws2gpart2
        #ws3
        ws3gpart2 = -1 * pm10[i+2]
        ws3g += part1 * ws3gpart2
        #ws4
        ws4gpart2 = -1 * pm10[i+3]
        ws4g += part1 * ws4gpart2
        #ws5
        ws5gpart2 = -1 * pm10[i+4]
        ws5g += part1 * ws5gpart2
        #ws6
        ws6gpart2 = -1 * pm10[i+5]
        ws6g += part1 * ws6gpart2
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
    adws1 += ws1g**2
    adws2 += ws2g**2
    adws3 += ws3g**2
    adws4 += ws4g**2
    adws5 += ws5g**2
    adws6 += ws6g**2
    adb += bg**2

    w1 -= lr / np.sqrt(adw1) * w1g
    w2 -= lr / np.sqrt(adw2) * w2g
    w3 -= lr / np.sqrt(adw3) * w3g
    w4 -= lr / np.sqrt(adw4) * w4g
    w5 -= lr / np.sqrt(adw5) * w5g
    w6 -= lr / np.sqrt(adw6) * w6g
    ws1 -= lr / np.sqrt(adws1) * ws1g
    ws2 -= lr / np.sqrt(adws2) * ws2g
    ws3 -= lr / np.sqrt(adws3) * ws3g
    ws4 -= lr / np.sqrt(adws4) * ws4g
    ws5 -= lr / np.sqrt(adws5) * ws5g
    ws6 -= lr / np.sqrt(adws6) * ws6g
    b -= lr / np.sqrt(adb) * bg

print(w1, w2, w3, w4, w5, w6, b)
print(ws1, ws2, ws3, ws4, ws5, ws6, b)
#exit


res=[]
variance = 0
model = 0
model_h = [0, 0, 0, 0, 0]
for i in rangeOfDataKey : 
    model = \
        w1 * pm25[i] + \
        w2 * pm25[i + 1] + \
        w3 * pm25[i + 2] + \
        w4 * pm25[i + 3] + \
        w5 * pm25[i + 4] + \
        w6 * pm25[i + 5] + \
        ws1 * pm10[i] + \
        ws2 * pm10[i + 1] + \
        ws3 * pm10[i + 2] + \
        ws4 * pm10[i + 3] + \
        ws5 * pm10[i + 4] + \
        ws6 * pm10[i + 5] + \
        b
    model_h.append(model)
    variance += (model - pm25[i + 1])**2

#x = range(len(loss_h))
#plt.plot(range(len(loss_h)), loss_h, label='loss')
plt.plot(range(len(model_h)), model_h, label='res')
plt.plot(range(len(pm25)), pm25, label='PM2.5')
#plt.plot(x, pm10[0:200], label='PM10')
#plt.plot(x, rain[0:200], label='Rain')
#plt.plot(x, so2[0:200], label='SO2')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple Plot")

plt.legend()

plt.show()




