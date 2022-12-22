# -*- coding utf-8 -*-
"""
Created on Sat Mar  5 05:51:15 2022

@author: onais
"""

import re
from tqdm import tqdm
import pandas as pd
import json 
#hg
DF=[]
ii=0
count = 0
FinalList=[]
# Strips the newline character
DF=pd.DataFrame()
C=1
CC=1
JsonData={}
AllAddress_Key_Value_As_MASK_Comp={}
USAD_Conversion_Dict={"1":"USAD_SNO","2":"USAD_SPR","3":"USAD_SNM","4":"USAD_SFX","5":"USAD_SPT","6":"USAD_ANM","7":"USAD_ANO","8":"USAD_CTY","9":"USAD_STA","10":"USAD_ZIP","11":"USAD_ZP4","12":"USAD_BNM","13":"USAD_BNO","14":"USAD_RNM"}
USAD_Conversion_Dict_Detail={"1":"USAD_SNO Street Number","2":"USAD_SPR Street Pre-directional","3":"USAD_SNM Street Name","4":"USAD_SFX Street Suffix","5":"USAD_SPT Street Post-directional","6":"USAD_ANM Secondary Address Name","7":"USAD_ANO Secondary Address Number","8":"USAD_CTY City Name","9":"USAD_STA State Name","10":"USAD_ZIP Zip Code","11":"USAD_ZP4 Zip 4 Code","12":"USAD_BNM Box Name","13":"USAD_BNO Box Number","14":"USAD_RNM Route Name"}
i=1
List_Index_Mask={}
with open('ExceptionFile.json','r+') as d:
    FirstPhaseLisCopyt=json.load(d)
    FirstPhaseList=FirstPhaseLisCopyt
    AddOrRemove=False
    MaskComparision={}
    for Key, Value in FirstPhaseLisCopyt.items():
        for Key1,Key2 in MaskComparision.items():
            if(Key==Key1):
                FirstPhaseList.pop(Key)
    Break=False
    for Key,Value in list(FirstPhaseList.items()):
        MaskI=1
        USAD_Mapping={"USAD_SNO":[],"USAD_SPR":[],"USAD_SNM":[],"USAD_SFX":[],"USAD_SPT":[],"USAD_ANM":[],"USAD_ANO":[],"USAD_CTY":[],"USAD_STA":[],"USAD_ZIP":[],"USAD_ZP4":[],"USAD_BNM":[],"USAD_BNO":[],"USAD_RNM":[]}
        Address=""
        for k in Value:
            for m,n in k.items():
                Address+=" "+n
    
        for k in Value:
            for k1,v1 in USAD_Conversion_Dict_Detail.items():
                print(v1,"=",k1)
            print("Address For Reference")
            print(Address)
            print(k)
            temp=(input("---> (Enter Index of Component)"))
            if temp not in USAD_Conversion_Dict_Detail.keys():
                Break=True
                break
            USAD_Mapping[USAD_Conversion_Dict[temp]].append(MaskI)
            MaskI+=1
        dict={k: v for k, v in USAD_Mapping.items() if v}
        print(dict)
        dictData={}
        dictData[Key]=dict
        Count_Of_Masks=0
        if Break:
            break
        with open('JSONMappingDefault.json', 'r+', encoding='utf-8') as f:
            data = json.load(f)
            Count_Of_Masks=len(data)+1
            # with open('Statistics.json', 'r+', encoding='utf-8') as g:
            #     Stat = json.load(g)
            #     Stat["Total_Mask_Count"]=Count_Of_Masks
            #     try:
            #         temp=Stat["Masks_Count"][Key]
            #         print(temp)
            #         Stat["Masks_Count"][Key]=temp+1
            #     except:
            #         Stat["Masks_Count"][Key]=1
            #     g.seek(0)
            #     json.dump(Stat,g,indent=4)
            #     g.truncate
            data[Key] =dict # <--- add `id` value.
            f.seek(0)        # <--- should reset file position to the beginning.
            json.dump(data, f)
            f.truncate()# remove remaining part
        FirstPhaseList.pop(Key)
        d.seek(0)        # <--- should reset file position to the beginning.
        json.dump(FirstPhaseList, d,indent=4)
        d.truncate()# remove re
    


        # print("Mask Generated is ",Mask_1)
    # print("Index\tMaskToken\t\tAddress Token")
    # i=1
    # for k in FirstPhaseList:
    #     for key,value in k.items():
    #         print(i,"\t\t",key,"\t\t\t\t",value) 
    #     i+=1