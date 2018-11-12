import csv
import time

f = open("data/train.csv")
readers = csv.reader(f)

keys = ('AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR')

dayOfContent = {}
components = {}

for rkey, row in enumerate(readers) :
    if(rkey > 0) :
        for cKey, column in enumerate(row) :
            if(cKey > 1) :
                timeString = '{} {}'.format(row[0], str(cKey - 2))
                ts = time.strptime(timeString, '%Y/%m/%d %H')
                tsKey = time.mktime(ts)
                components[row[1]] = column
        dayOfContent[str(int(tsKey))] = components

print(dayOfContent)
                

