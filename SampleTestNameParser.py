import re
from tqdm import tqdm
import pandas as pd
import json 
#Parsing 1st program
Name_4CAF50=open("Names.txt","r")
Lines = Name_4CAF50.readlines()
DF=[]
ii=0
count = 0
FinalList=[]
fileHandle = open('NamesWordTable.txt', 'r')
# Strips the newline character
Count=len(Lines)
DF=pd.DataFrame()
C=1
CC=1
JsonData={}
AllName_Key_Value_As_MASK_Comp={}
Observation=0
Total=0
dataFinal={}
with open('ConvertedNamesOutput.json', 'r+', encoding='utf-8') as M:
    dataFinal= json.load(M)

    for line in tqdm(Lines):
        line=line.strip("\n").split("|")
        ID=line[0]
        line=line[1] .strip() 
        Old_Name=line.strip()
        Names_Conversion_Dict={"1":"Prefix Title","2":"Given Name", "3":"Surname","4" :"Generational Suffix", "5":"Suffix Title"}
        USAD_Conversion_Dict_Detail={"1":"USAD_SNO Street Number","2":"USAD_SPR Street Pre-directional","3":"USAD_SNM Street Name","4":"USAD_SFX Street Suffix","5":"USAD_SPT Street Post-directional","6":"USAD_ANM Secondary Name Name","7":"USAD_ANO Secondary Name Number","8":"USAD_CTY City Name","9":"USAD_STA State Name","10":"USAD_ZIP Zip Code","11":"USAD_ZP4 Zip 4 Code","12":"USAD_BNM Box Name","13":"USAD_BNO Box Number","14":"USAD_RNM Route Name"}
    
        
        List=Names_Conversion_Dict.keys()
        FirstPhaseList=[]
        Name=re.sub(',',' , ',line)
        Name=re.sub(' +', ' ',Name)
        Name=re.sub('[.]','',Name)
        #Name=re.sub('#','',Name)    
        Name=Name.upper()
        NameList = re.split("\s|\s,\s ", Name)
        #del(NameList[len(NameList)-1])
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
        Names_Conversion_Mapping={"Prefix Title":[],"Given Name":[], "Surname":[],"Generational Suffix":[],"Suffix Title":[]}
        Start=0
        Counts=0
        FirstPhase_WithComma=FirstPhaseList
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
        print(Found)
        if Found:
            Observation+=1
            AllName_Key_Value_As_MASK_Comp={}
            Mappings={}
            for K2,V2 in FoundDict[Mask_1].items():
                Temp=""
                for p in V2:
                    for K3,V3 in FirstPhaseList[p-1].items():
                       Temp+=" "+V3
                       Temp=Temp.strip()
                       Mappings[K2]=Temp
            try:
                if dataFinal[Mask_1][ID]:
                    continue
            except:
                a=0
            try:
                dataFinal[Mask_1][ID] =Mappings # <--- add `id` value.
            except:
                dataFinal[Mask_1]={}
                dataFinal[Mask_1][ID]=Mappings
            with open('JSONMappingDefault.json', 'r+', encoding='utf-8') as f:
                data = json.load(f)
                Count_Of_Masks=len(data)+1
                with open('Statistics.json', 'r+', encoding='utf-8') as g:
                    Stat = json.load(g)
                    Stat["Total_Mask_Count"]=Count_Of_Masks
                    try:
                        temp=Stat["Masks_Count"][Mask_1]
                        Stat["Masks_Count"][Mask_1]=temp+1
                    except:
                        Stat["Masks_Count"][Mask_1]=1
                    g.seek(0)
                    json.dump(Stat,g,indent=4)
                    g.truncate
            
        elif not FoundExcept:  
            with open('NameExceptionFile.json', 'r+', encoding='utf-8') as g:
                Stat = json.load(g)
                Stat[Mask_1]=FirstPhaseList
                g.seek(0)
                json.dump(Stat,g,indent=4)
                g.truncate
        Total+=1
    M.seek(0)
    json.dump(dataFinal, M,indent=4)
    M.truncate()
    
print("Name Matching Report")
print("Total=",Count)
print("Matched Namees=",Observation)
print("Percentage of Matched",(Observation/Count)*100)
