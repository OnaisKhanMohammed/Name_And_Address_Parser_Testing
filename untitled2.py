# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 01:15:11 2023

@author: onais
"""

file=open("split.txt","r",encoding='utf8')
out=file.readlines()
for m in out:
    print(m.split(" ")[0])