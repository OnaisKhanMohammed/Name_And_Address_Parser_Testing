# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 13:32:17 2022

@author: onais
"""
import re
import collections 
import json 
#Parsing 1st program

def ExtractNames(File,TruthSet):
    FinalMappings={}
    Truth_Result={}
    Name_4CAF50=open(File,"r")
    Lines = Name_4CAF50.readlines()
    fileHandle = open('NamesWordTableOpt.txt', 'r')
    # Strips the newline character
    Observation=0
    Total=0
    for line in Lines:
        Total+=1
        Names_Conversion_Dict={"1":"Prefix Title","2":"Given Name", "3":"Surname","4" :"Generational Suffix", "5":"Suffix Title"}    
        Nm=line.split("|")
        line=Nm[1]
        ID=Nm[0]
        FirstPhaseList=[]
        Name=re.sub(',',' , ',line)
        Name=re.sub(' +', ' ',Name)
        Name=re.sub(' +', ' ',Name)
        Name=re.sub('[.]','',Name)
        #Name=re.sub('#','',Name)    
        Name=Name.upper()
        NameList = re.split("\s|\s,\s ", Name)
        NameList = ' '.join(NameList).split()
        TrackKey=[]
        Mask=[]
        Combine=""
        LoopCheck=1
        for A in NameList:
            FirstPhaseDict={}
            NResult=False
            if A==",":
                O=0
                Mask.append(Combine)
                Combine=""
                FirstPhaseList.append(",")
                #FirstPhaseList.append("Seperator")
            elif A==" ":
                continue
            elif A!="," and len(A)==1:
                Combine+="I"
                TrackKey.append("I")
                FirstPhaseDict["I"] = A
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
                        break
                if NResult==False:
                    Combine+="W"
                    TrackKey.append("W")
                    FirstPhaseDict["W"] = A
                    FirstPhaseList.append(FirstPhaseDict)
            if LoopCheck==len(NameList):
                Mask.append(Combine)
            fileHandle.seek(0)
            LoopCheck+=1
        Mask_1=",".join(Mask)
        FirstPhaseList = [FirstPhaseList[b] for b in range(len(FirstPhaseList)) if FirstPhaseList[b] != ","]
        data={}
        with open('JSONMappingNameDefault.json', 'r+', encoding='utf-8') as f:
            data = json.load(f)
        Found=False
        FoundDict={}
        for tk,tv in data.items():
            if(tk==Mask_1):
                FoundDict[tk]=tv
                Found=True
                break
        FoundExcept=False
        with open('NameExceptionFile.json', 'r+', encoding='utf-8') as g:
            Stat = json.load(g)
            if Mask_1 in Stat.keys():
                FoundExcept=True
        Mappings={}
        if Found:
            Truth_Result[ID]=FoundDict[Mask_1]
            Observation+=1
            for K2,V2 in FoundDict[Mask_1].items():
                Temp=""
                for p in V2:
                    for K3,V3 in FirstPhaseList[p-1].items():
                       Temp+=" "+V3
                       Temp=Temp.strip()
                       Mappings[K2]=Temp
            FinalMappings[ID]=Mappings
        elif not FoundExcept:  
            with open('NameExceptionFile.json', 'r+', encoding='utf-8') as g:
                Stat = json.load(g)
                Stat[Mask_1]=FirstPhaseList
                g.seek(0)
                json.dump(Stat,g,indent=4)
                g.truncate
    Result={}      
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
    print(FinalMappings)
    return (FinalMappings,(Observation/Total)*100,Count_of_Correct/Total_Count*100)
        
    
    
    
