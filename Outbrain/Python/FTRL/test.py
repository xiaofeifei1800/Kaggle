# -*- coding: utf-8 -*-
"""
Thanks to tinrtgu for the wonderful base script
Use pypy for faster computations.!
"""
import csv
import sys
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
import pandas as pd
csv.field_size_limit(sys.maxsize)

# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
######################################################-66########################

# A, paths
data_path = "/Users/xiaofeifei/I/Kaggle/Outbrain/"


# C, feature/hash trick
D = 2 ** 20             # number of weights to use
interaction = None     # whether to enable poly2 f gleature interactions


start = datetime.now()

print("document..")
with open("/Users/xiaofeifei/GitHub/Kaggle/Outbrain/Python/all_train.csv") as infile:
    document = csv.reader(infile)
    #prcont_header = (prcont.next())[1:]
    document_header = next(document)[1:]
    document_dict = {}
    year =[]
    month = []
    day = []
    hour = []
    for ind,row in enumerate(document):

        date = row[14]
        if date == 0:
            year.append(0)
            month.append(0)
            day.append(0)
            hour.append(0)
        else:
            year.append(date[0:4])
            month.append(date[5:7])
            day.append(date[8:10])
            hour.append(date[11:13])
        if ind%100000 == 0:
            print("document file : ", ind)
        if ind==1000:
            break
    print(len(document_dict))
print year
del document


