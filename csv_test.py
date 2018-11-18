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

x = range(200)
pm25 = getValue('PM2.5')
pm10 = getValue('PM10')
rain = getValue('RAINFALL')
so2  = getValue('SO2')

plt.plot(x, pm25[0:200], label='PM2.5')
plt.plot(x, pm10[0:200], label='PM10')
plt.plot(x, rain[0:200], label='Rain')
plt.plot(x, so2[0:200], label='SO2')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple Plot")

plt.legend()

plt.show()



