# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 00:06:20 2022

@author: onais
"""


import re
from tqdm import tqdm
import pandas as pd
import json 
import collections 
#Parsing 1st program


def Address_Parser(line):
    Result={}
    fileHandle = open('USAddressWordTable.txt', 'r')
    # Strips the newline character
    Observation=0
    Total=0
    Truth_Result={}
    dataFinal={}
    FirstPhaseList=[]
    Address=re.sub(',',' , ',line)
    Address=re.sub(' +', ' ',Address)
    Address=re.sub('[.]','',Address)
    #Address=re.sub('#','',Address)    
    Address=Address.upper()
    AddressList = re.split("\s|\s,\s ", Address)
    #del(AddressList[len(AddressList)-1])
    TrackKey=[]
    Mask=[]
    Combine=""
    LoopCheck=1
    for A in AddressList:
        FirstPhaseDict={}
        NResult=False
        try:
            Compare=A[0].isdigit()
        except:
            print()
        if A==",":
            Mask.append(Combine)
            Combine=""
            FirstPhaseList.append(",")
            #FirstPhaseList.append("Seperator")
        elif Compare:
            Combine+="N"
            TrackKey.append("N")
            FirstPhaseDict["N"]=A
            FirstPhaseList.append(FirstPhaseDict)
        else:
            for line in fileHandle:
                fields=line.split('|')
                if A==(fields[0]):
                    NResult=True
                    temp=fields[1]
                    Combine+=temp[0]
                    FirstPhaseDict[temp[0]] = A
                    FirstPhaseList.append(FirstPhaseDict)
                    TrackKey.append(temp[0])
            if NResult==False:
                Combine+="W"
                TrackKey.append("W")
                FirstPhaseDict["W"] = A
                FirstPhaseList.append(FirstPhaseDict)
        if LoopCheck==len(AddressList):
            Mask.append(Combine)
        fileHandle.seek(0)
        LoopCheck+=1
    Mask_1=",".join(Mask)
    FirstPhaseList = [FirstPhaseList[b] for b in range(len(FirstPhaseList)) if FirstPhaseList[b] != ","]
    data={}
    with open('JSONMappingDefault.json', 'r+', encoding='utf-8') as f:
        data = json.load(f)
    Found=False
    FoundDict={}
    for tk,tv in data.items():
        if(tk==Mask_1):
            FoundDict[tk]=tv
            Found=True
            break
    FoundExcept=False
    with open('ExceptionFile.json', 'r+', encoding='utf-8') as g:
        Stat = json.load(g)
        if Mask_1 in Stat.keys():
            FoundExcept=True
    if Found:
        Observation+=1
        Mappings={}
        for K2,V2 in FoundDict[Mask_1].items():
            Temp=""
            for p in V2:
                for K3,V3 in FirstPhaseList[p-1].items():
                   Temp+=" "+V3
                   Temp=Temp.strip()
                   Mappings[K2]=Temp
     
        try:
            Result["Output"]=Mappings
        except:
            Result["Output"]=Mappings
            
        
    elif not FoundExcept:  
        with open('ExceptionFile.json', 'r+', encoding='utf-8') as g:
            Stat = json.load(g)
            Stat[Mask_1]=FirstPhaseList
            g.seek(0)
            json.dump(Stat,g,indent=4)
            g.truncate
    Total+=1
    return Result
# print("Final Correct Address Parsing Percentage",Count_of_Correct/Total_Count*100)
# print("Address Matching Report")
# print("Total=",Count)
# print("Matched Addresses=",Observation)
# print("Percentage of Matched",(Observation/Count)*100)
        # print("Mask Generated is ",Mask_1)
    # print("Index\tMaskToken\t\tAddress Token")
    # i=1
    # for k in FirstPhaseList:
    #     for key,value in k.items():
    #         print(i,"\t\t",key,"\t\t\t\t",value) 
    #     i+=1