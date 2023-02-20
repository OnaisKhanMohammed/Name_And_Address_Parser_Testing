# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 12:32:34 2023

@author: onais
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 09:15:34 2022

@author: onais
"""

import re
from tqdm import tqdm
import AddressParserProject as RuleBased
import pandas as pd
import json 
import collections 
import PreprocessingNameAddress as PreProc
import sklearn
from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix,classification_report
#Parsing 1st program
import warnings
warnings.filterwarnings("ignore")

def Address_Parser(Address_4CAF50,TruthSet):
    Result={}
    RuleBasedOutput={}
    Exception_Mask=""
    Detailed_Report=""
    Address_4CAF50=open(Address_4CAF50,"r")
    Lines = Address_4CAF50.readlines()
    fileHandle = open('USAddressWordTable.txt', 'r')
    # Strips the newline character
    Observation=0
    Total=0
    Truth_Result={}
    dataFinal={} 
    USAD_Conversion_Dict={
  "USAD_SNO": 1,
  "USAD_SPR": 2,
  "USAD_SNM": 3,
  "USAD_SFX": 4,
  "USAD_SPT": 5,
  "USAD_ANM": 6,
  "USAD_ANO": 7,
  "USAD_CTY": 8,
  "USAD_STA": 9,
  "USAD_ZIP": 10,
  "USAD_ZP4": 11,
  "USAD_BNM": 12,
  "USAD_BNO": 13,
  "USAD_RNM": 14,
  "USAD_RNO": 15
}
    
    USAD_CONVERSION_={
        
  "1": "USAD_SNO",
  "2": "USAD_SPR",
  "3": "USAD_SNM",
  "4": "USAD_SFX",
  "5": "USAD_SPT",
  "6": "USAD_ANM",
  "7": "USAD_ANO",
  "8": "USAD_CTY",
  "9": "USAD_STA",
  "10": "USAD_ZIP",
  "11": "USAD_ZP4",
  "12": "USAD_BNM",
  "13": "USAD_BNO",
  "14": "USAD_RNM",
  "15": "USAD_RNO"
}
    with open('ConvertedJSONAddressesOutput.json', 'r+', encoding='utf-8') as M:
        dataFinal= json.load(M)
        Detailed_Report+="Exception and Mask Report\n"
        for line in tqdm(Lines):
            line=line.strip("\n").split("|")
            ID=line[0].strip()
            line=line[1].strip()
            Address=line
            FirstPhaseList=[]
            PackAddress=PreProc.PreProcessingNameAddress().AddresssCleaning(line)
            AddressList=PackAddress[0]
            AddressList = [i for i in AddressList if i]
          
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
                    Truth_Result[ID]=Mappings
                    Result[ID]=Mappings
                    dataFinal[Mask_1][ID] =Mappings # <--- add `id` value.
                except:
                    Result[ID]=Mappings
                    Truth_Result[ID]=Mappings
                    dataFinal[Mask_1]={}
                    dataFinal[Mask_1][ID]=Mappings
                
            elif not FoundExcept:  
                with open('ExceptionFile.json', 'r+', encoding='utf-8') as g:
                    Stat = json.load(g)
                    Stat[Mask_1]=FirstPhaseList
                    g.seek(0)
                    json.dump(Stat,g,indent=4)
                    g.truncate
                    RuleBasedOutput[ID]=RuleBased.RuleBasedAddressParser.AddressParser(Address)
            else:
                RuleBasedOutput[ID]=RuleBased.RuleBasedAddressParser.AddressParser(Address)
                Exception_Mask+=Mask_1+"\n"
            
            Total+=1
        M.seek(0)
        json.dump(dataFinal, M,indent=4)
        M.truncate()


    
    Count_of_Correct=0
    Total_Count=0  
    y_test=[]
    y_predict=[]
    open("Individual Address1_Truth_File.txt","r")
    with open("Individual Address1_Truth_File.txt", 'r+', encoding='utf-8') as g:

        Stat = json.load(g)
        Count_of_Correct=0
        Total_Count=0
                
        for key,value in Truth_Result.items():
            Total_Count=len(Truth_Result)

            if key in Stat.keys():
                Count1=0
                Count_total=0
                for k1,v1 in value.items():
                    predict=False
                    y_test.append(USAD_Conversion_Dict[k1])
                    for k2,v2 in Stat[key].items():
                        if v1==v2:
                            y_predict.append(USAD_Conversion_Dict[k2])
                            Count1+=len(Stat[key])
                            predict=True
                            break
           
                    if not predict:
                        for f in v1.split(" "):
                            for k2,v2 in Stat[key].items():
                                if f in v2:
                                    y_predict.append(USAD_Conversion_Dict[k2])
                                    predict=True
                                    break
                            if predict:
                                break
                            else:
                                y_predict.append(0)

    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    Confusion =  multilabel_confusion_matrix(y_test, y_predict)
    df=classification_report(y_test,y_predict,output_dict=True)
    df_report = pd.DataFrame(df).transpose()

    df_report.reset_index(inplace=True)
    df_report=df_report.replace({"index": USAD_CONVERSION_})
    df_report[1:].to_csv("Metrics.csv")
    
    RTruth=0
    try:
        RTruth=(Count_of_Correct/Total_Count*100)
    except:
        print()
    Detailed_Report+="Output From Active Learning\n\n"
    ActiveLResult = json.dumps(Result, indent = 4) 
    Detailed_Report+=str(ActiveLResult)
    RuleBasedRes =json.dumps(RuleBasedOutput,indent=4)
    Detailed_Report+="\n\nOutput Fron Rule Based Approach\n\n"
    Detailed_Report+=str(RuleBasedRes)
    Detailed_Report+="\n\nNumber of Exceptions Thrown: -\t"+str(Total-Observation)+"\n"
    Detailed_Report+="Number of Parsed Address: -\t"+str(Observation)+"\n"
    Detailed_Report+="Percentage of Parsed Result: -\t"+str((Observation/Total)*100)+"\n"
    Detailed_Report+="List of Exception Mask(s): -\t\n\n"+Exception_Mask+"--"
    Detailed_Report+="\n\n Evaluation Metrics\n\n"

    Detailed_Report+=str(df_report[1:])
    f=open("Detailed_Report.txt","w")

    f.write(Detailed_Report)
    f.close()

    
    
    return (Result,(Observation/Total)*100,RTruth)
Address_Parser("Individual Address1_Input_File.txt","Individual Address1_Truth_File.txt")

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