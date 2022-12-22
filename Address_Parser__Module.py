# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 09:15:34 2022

@author: onais
"""

import re
from tqdm import tqdm
import pandas as pd
import json 
import collections 
#Parsing 1st program


def Address_Parser(Address_4CAF50,TruthSet):
    Result={}
    Address_4CAF50=open(Address_4CAF50,"r")
    Lines = Address_4CAF50.readlines()
    fileHandle = open('USAddressWordTable.txt', 'r')
    # Strips the newline character
    Observation=0
    Total=0
    Truth_Result={}
    dataFinal={}
    with open('ConvertedJSONAddressesOutput.json', 'r+', encoding='utf-8') as M:
        dataFinal= json.load(M)
    
        for line in tqdm(Lines):
            line=line.strip("\n").split("|")
            ID=line[0]
            line=line[1] .strip() 
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
                Truth_Result[ID]=FoundDict[Mask_1]
                for K2,V2 in FoundDict[Mask_1].items():
                    Temp=""
                    for p in V2:
                        for K3,V3 in FirstPhaseList[p-1].items():
                           Temp+=" "+V3
                           Temp=Temp.strip()
                           Mappings[K2]=Temp
             
                try:
                    Result[ID]=Mappings
                    dataFinal[Mask_1][ID] =Mappings # <--- add `id` value.
                except:
                    Result[ID]=Mappings
                    dataFinal[Mask_1]={}
                    dataFinal[Mask_1][ID]=Mappings
                
            elif not FoundExcept:  
                with open('ExceptionFile.json', 'r+', encoding='utf-8') as g:
                    Stat = json.load(g)
                    Stat[Mask_1]=FirstPhaseList
                    g.seek(0)
                    json.dump(Stat,g,indent=4)
                    g.truncate
            Total+=1
        M.seek(0)
        json.dump(dataFinal, M,indent=4)
        M.truncate()
    with open(TruthSet, 'r+', encoding='utf-8') as g:
        Stat = json.load(g)
        Count_of_Correct=0
        Total_Count=0
        for key,value in Truth_Result.items():
            Total_Count=len(Truth_Result)
            if key in Stat.keys():
                Count1=0
                Count_total=0
                for k1,v1 in value.items():
                    Count_total+=len(Stat[key])
                    for k2,v2 in Stat[key].items():
                        if collections.Counter(value[k1]) == collections.Counter(Stat[key][k2]) and k1==k2:
                            Count1+=len(Stat[key]) 
                print("ID:",key, "Percentage of Correctness",round((Count1/Count_total)*100,2),"%")
                if (round((Count1/Count_total)*100))>99:
                          Count_of_Correct+=1
        return (Result,(Observation/Total)*100,Count_of_Correct/Total_Count*100)
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