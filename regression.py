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
NOx  = getValue('NOx')
NO2  = getValue('NO2')

comps = defaultdict(dict)
#for comp in keys :
#for comp in ('PM2.5', 'NOx', 'NO2', 'PM10') :
#    comps[comp] = getValue(comp)

rangeOfDataKey = range(len(pm25) - 6)
w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = w9 = wa = wb = wc = wd = we = wf = wg = \
wh = wi = wj = wk = wl = wm = wn = wo =wp = b = 1

adgw1 = adgw2 = adgw3 = adgw4 = adgw5 = adgw6 = adgw7 = adgw8 = adgw9 = adgwa = \
adgwb = adgwc = adgwd = adgwe = adgwf = adgwg = adgwh = adgwi = adgwj = adgwk = \
adgwl = adgwm = adgwn = adgwo = adgwp = adgb = 0

lost = 0.0
lr = 1000
tLoop = 1000

for ss in range(tLoop) :
    for loopOfDataKey in rangeOfDataKey :

        x1 = pm25[loopOfDataKey]
        x2 = pm25[loopOfDataKey + 1]
        x3 = pm25[loopOfDataKey + 2]
        x4 = pm25[loopOfDataKey + 3]
        x5 = pm25[loopOfDataKey + 4]
 
        x6 = pm10[loopOfDataKey]
        x7 = pm10[loopOfDataKey + 1]
        x8 = pm10[loopOfDataKey + 2]
        x9 = pm10[loopOfDataKey + 3]
        xa = pm10[loopOfDataKey + 4]

        xb = NOx[loopOfDataKey]
        xc = NOx[loopOfDataKey + 1]
        xd = NOx[loopOfDataKey + 2]
        xe = NOx[loopOfDataKey + 3]
        xf = NOx[loopOfDataKey + 4]
    
        xh = NO2[loopOfDataKey]
        xi = NO2[loopOfDataKey + 1]
        xj = NO2[loopOfDataKey + 2]
        xk = NO2[loopOfDataKey + 3]
        xl = NO2[loopOfDataKey + 4]

        # x1 x2 x3 x4 x5 五次式
        #y_prod = \
        #w1 * x1**5 + w2 * x1**4 + w3 * x1**3 + w4 * x1**2 + w5 * x1 + \
        #w6 * x2**5 + w7 * x2**4 + w8 * x2**3 + w9 * x2**2 + wa * x2 + \
        #wb * x3**5 + wc * x3**4 + wd * x3**3 + we * x3**2 + wf * x3 + \
        #wg * x4**5 + wh * x4**4 + wi * x4**3 + wj * x4**2 + wk * x4 + \
        #wl * x5**5 + wm * x5**4 + wn * x5**3 + wo * x5**2 + wp * x5 + b

        # x1 x2 x3 x4 x5 三次式
        y_prod = \
        w3 * x1**3 + w4 * x1**2 + w5 * x1 + \
        w8 * x2**3 + w9 * x2**2 + wa * x2 + \
        wd * x3**3 + we * x3**2 + wf * x3 + \
        wi * x4**3 + wj * x4**2 + wk * x4 + \
        wn * x5**3 + wo * x5**2 + wp * x5 + b
        
        y_prod = \
        w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7 + \
        w8 * x8 + w9 * x9 + wa * xa + wb * xb + wc * xc + wd * xd + we * xe + \
        wf * xf + wh * xh + wi * xi + wj * xj + wk * xk + wl * xl + b

        lost = lost + np.square(pm25[loopOfDataKey + 5] - y_prod)

        lOfW1 = (pm25[loopOfDataKey + 5] - y_prod) * -2 * x1
        lOfW2 = (pm25[loopOfDataKey + 5] - y_prod) * -2 * x2
        lOfW3 = (pm25[loopOfDataKey + 5] - y_prod) * -2 * x3
        lOfW4 = (pm25[loopOfDataKey + 5] - y_prod) * -2 * x4
        lOfW5 = (pm25[loopOfDataKey + 5] - y_prod) * -2 * x5
        
        lOfW6 = (pm25[loopOfDataKey + 5] - y_prod) * -2 * x6
        lOfW7 = (pm25[loopOfDataKey + 5] - y_prod) * -2 * x7
        lOfW8 = (pm25[loopOfDataKey + 5] - y_prod) * -2 * x8
        lOfW9 = (pm25[loopOfDataKey + 5] - y_prod) * -2 * x9
        lOfWa = (pm25[loopOfDataKey + 5] - y_prod) * -2 * xa
        
        lOfWb = (pm25[loopOfDataKey + 5] - y_prod) * -2 * xb
        lOfWc = (pm25[loopOfDataKey + 5] - y_prod) * -2 * xc
        lOfWd = (pm25[loopOfDataKey + 5] - y_prod) * -2 * xd
        lOfWe = (pm25[loopOfDataKey + 5] - y_prod) * -2 * xe
        lOfWf = (pm25[loopOfDataKey + 5] - y_prod) * -2 * xf
        
        #lOfWg = (pm25[loopOfDataKey + 5] - y_prod) * -2 * xg
        lOfWh = (pm25[loopOfDataKey + 5] - y_prod) * -2 * xh
        lOfWi = (pm25[loopOfDataKey + 5] - y_prod) * -2 * xi
        lOfWj = (pm25[loopOfDataKey + 5] - y_prod) * -2 * xj
        lOfWk = (pm25[loopOfDataKey + 5] - y_prod) * -2 * xk
        
        lOfWl = (pm25[loopOfDataKey + 5] - y_prod) * -2 * xl
        #lOfWm = (pm25[loopOfDataKey + 5] - y_prod) * -2 * x5**4
        #lOfWn = (pm25[loopOfDataKey + 5] - y_prod) * -2 * x5**3
        #lOfWo = (pm25[loopOfDataKey + 5] - y_prod) * -2 * x5**2
        #lOfWp = (pm25[loopOfDataKey + 5] - y_prod) * -2 * x5
        
        lOfb  = (pm25[loopOfDataKey + 5] - y_prod) * -2

        adgw1 = adgw1 + lOfW1**2
        adgw2 = adgw2 + lOfW2**2
        adgw3 = adgw3 + lOfW3**2
        adgw4 = adgw4 + lOfW4**2
        adgw5 = adgw5 + lOfW5**2
        adgw6 = adgw6 + lOfW6**2
        adgw7 = adgw7 + lOfW7**2
        adgw8 = adgw8 + lOfW8**2
        adgw9 = adgw9 + lOfW9**2
        adgwa = adgwa + lOfWa**2
        adgwb = adgwb + lOfWb**2
        adgwc = adgwc + lOfWc**2
        adgwd = adgwd + lOfWd**2
        adgwe = adgwe + lOfWe**2
        adgwf = adgwf + lOfWf**2
        #adgwg = adgwg + lOfWg**2
        adgwh = adgwh + lOfWh**2
        adgwi = adgwi + lOfWi**2
        adgwj = adgwj + lOfWj**2
        adgwk = adgwk + lOfWk**2
        adgwl = adgwl + lOfWl**2
        #adgwm = adgwm + lOfWm**2
        #adgwn = adgwn + lOfWn**2
        #adgwo = adgwo + lOfWo**2
        #adgwp = adgwp + lOfWp**2
        adgb  = adgb  + lOfb**2
        
    w1 = w1 - lr * lOfW1 / np.sqrt(adgw1)
    w2 = w2 - lr * lOfW2 / np.sqrt(adgw2)
    w3 = w3 - lr * lOfW3 / np.sqrt(adgw3)
    w4 = w4 - lr * lOfW4 / np.sqrt(adgw4)
    w5 = w5 - lr * lOfW5 / np.sqrt(adgw5)
    w6 = w6 - lr * lOfW6 / np.sqrt(adgw6)
    w7 = w7 - lr * lOfW7 / np.sqrt(adgw7)
    w8 = w8 - lr * lOfW8 / np.sqrt(adgw8)
    w9 = w9 - lr * lOfW9 / np.sqrt(adgw9)
    wa = wa - lr * lOfWa / np.sqrt(adgwa)
    wb = wb - lr * lOfWb / np.sqrt(adgwb)
    wc = wc - lr * lOfWc / np.sqrt(adgwc)
    wd = wd - lr * lOfWd / np.sqrt(adgwd)
    we = we - lr * lOfWe / np.sqrt(adgwe)
    wf = wf - lr * lOfWf / np.sqrt(adgwf)
    #wg = wg - lr * lOfWg / np.sqrt(adgwg)
    wh = wh - lr * lOfWh / np.sqrt(adgwh)
    wi = wi - lr * lOfWi / np.sqrt(adgwi)
    wj = wj - lr * lOfWj / np.sqrt(adgwj)
    wk = wk - lr * lOfWk / np.sqrt(adgwk)
    wl = wl - lr * lOfWl / np.sqrt(adgwl)
    #wm = wm - lr * lOfWm / np.sqrt(adgwm)
    #wn = wn - lr * lOfWn / np.sqrt(adgwn)
    #wo = wo - lr * lOfWo / np.sqrt(adgwo)
    #wp = wp - lr * lOfWp / np.sqrt(adgwp)
    b  = b  - lr * lOfb  / np.sqrt(adgb)

    print("==============")
    #print(w1, w2, w3, w4, w5, w6, w7, w8, w9, wa, wb, wc, wd, we, wf, wg, wh, wi, wj, wk, wl, wm, wn, wo, wp, b)
    print(w1, w2, w3, w4, w5, w6, w7, w8, w9, wa, wb, wc, wd, we, wf, wh, wi, wj, wk, wl, b)
    #print(w3, w4, w5, w8, w9, wa, wd, we, wf, wi, wj, wk, wn, wo, wp, b)
    print("--------------")
    print(lost)
    lost = 0
    
sys.exit
x = range(200)

for keyComps, valueComps in comps.items() :
    plt.plot(x, valueComps[0:200], label=keyComps)


plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple Plot")

plt.legend()

plt.show()




