import re
from tqdm import tqdm
import pandas as pd
import json 
import collections 
#Parsing 1st program
Address_4CAF50=open("AddressListRev.txt","r")
Lines = Address_4CAF50.readlines()
DF=[]
ii=0
count = 0
FinalList=[]
fileHandle = open('USAddressWordTable.txt', 'r')
# Strips the newline character
Count=len(Lines)
DF=pd.DataFrame()
C=1
CC=1
JsonData={}
AllAddress_Key_Value_As_MASK_Comp={}
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
        Old_Address=line.strip()
        USAD_Conversion_Dict={"1":"USAD_SNO","2":"USAD_SPR","3":"USAD_SNM","4":"USAD_SFX","5":"USAD_SPT","6":"USAD_ANM","7":"USAD_ANO","8":"USAD_CTY","9":"USAD_STA","10":"USAD_ZIP","11":"USAD_ZP4","12":"USAD_BNM","13":"USAD_BNO","14":"USAD_RNM"}
        
        USAD_Conversion_Dict_Detail={"1":"USAD_SNO Street Number","2":"USAD_SPR Street Pre-directional","3":"USAD_SNM Street Name","4":"USAD_SFX Street Suffix","5":"USAD_SPT Street Post-directional","6":"USAD_ANM Secondary Address Name","7":"USAD_ANO Secondary Address Number","8":"USAD_CTY City Name","9":"USAD_STA State Name","10":"USAD_ZIP Zip Code","11":"USAD_ZP4 Zip 4 Code","12":". Box Name","13":"USAD_BNO Box Number","14":"USAD_RNM Route Name"}

        
        List=USAD_Conversion_Dict.keys()
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
                O=0
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
        USAD_Mapping={"USAD_SNO":[],"USAD_SPR":[],"USAD_SPR":[],"USAD_SNM":[],"USAD_SFX":[],"USAD_SPT":[],"USAD_ANM":[],"USAD_ANO":[],"USAD_CTY":[],"USAD_STA":[],"USAD_ZIP":[],"USAD_ZP4":[],"USAD_BNM":[],"USAD_BNO":[],"USAD_RNM":[]}
        Start=0
        Counts=0
        FirstPhase_WithComma=FirstPhaseList
        FirstPhaseList = [FirstPhaseList[b] for b in range(len(FirstPhaseList)) if FirstPhaseList[b] != ","]
        data={}
        with open('JSONMAPPING-DummyFile.json', 'r+', encoding='utf-8') as f:
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
            AllAddress_Key_Value_As_MASK_Comp={}
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
                
                if dataFinal[Mask_1][ID]:
                    
                    continue
            except:
                a=0
            try:
                dataFinal[Mask_1][ID] =Mappings # <--- add `id` value.
            except:
                dataFinal[Mask_1]={}
                dataFinal[Mask_1][ID]=Mappings
            with open('JSONMAPPING-DummyFile.json', 'r+', encoding='utf-8') as f:
                data = json.load(f)
                Count_Of_Masks=len(data)+1
                # with open('Statistics.json', 'r+', encoding='utf-8') as g:
                #     Stat = json.load(g)
                #     Stat["Total_Mask_Count"]=Count_Of_Masks
                #     try:
                #         temp=Stat["Masks_Count"][Mask_1]
                #         Stat["Masks_Count"][Mask_1]=temp+1
                #     except:
                #         Stat["Masks_Count"][Mask_1]=1
                #     g.seek(0)
                #     json.dump(Stat,g,indent=4)
                #     g.truncate
            
            
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
print("\n")
with open('AddressTruthFile.txt', 'r+', encoding='utf-8') as g:
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
print("Final Correct Address Parsing Percentage",Count_of_Correct/Total_Count*100)
print("Address Matching Report")
print("Total=",Count)
print("Matched Addresses=",Observation)
print("Percentage of Matched",(Observation/Count)*100)
        # print("Mask Generated is ",Mask_1)
    # print("Index\tMaskToken\t\tAddress Token")
    # i=1
    # for k in FirstPhaseList:
    #     for key,value in k.items():
    #         print(i,"\t\t",key,"\t\t\t\t",value) 
    #     i+=1