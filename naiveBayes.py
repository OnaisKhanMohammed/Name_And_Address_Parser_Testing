# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 02:13:44 2022

@author: onais
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# import re
# from tqdm import tqdm
# import pandas as pd
# import json 
# df = pd.read_csv('Merged_name_address.txt', names=['product_id','product_title','category_label'])

# Address_4CAF50=open("Merged_name_address.txt","r")
# Lines = Address_4CAF50.readlines()
# # Preprocessing the Data
# TrainingData=pd.DataFrame()
# for line in tqdm(Lines):
#     line=line.strip("\n").split("|")
#     ID=line[0]
#     line=line[1] .strip() 
#     FirstPhaseList=[]
#     Address=re.sub(',',' , ',line)
#     Address=re.sub(' +', ' ',Address)
#     Address=re.sub('[.]','',Address)
#     #Address=re.sub('#','',Address)    
#     Address=Address.upper()
#     AddressList = re.split("\s|\s,\s ", Address)
#     #Processing Training Data
#     for m in AddressList:
        
df = pd.read_csv('TestNaive.csv', names=['TOKEN','RESULT'])
count_vec = TfidfVectorizer()
bow = count_vec.fit_transform(df['TOKEN'])
bow = np.array(bow.todense())
X = bow
y =np.array( df['RESULT'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model = MultinomialNB().fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

