import csv
import time

f = open("data/train.csv")
    readers = csv.reader(f)
keys = ('AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR')

item = defaultdict(dict)

item['a']['c'] = '123'

print(item)
                

