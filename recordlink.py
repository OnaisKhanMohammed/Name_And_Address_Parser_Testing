# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:45:09 2023

@author: onais
"""

import recordlinkage

import pandas as pd
import recordlinkage

from recordlinkage.datasets import load_febrl4

print(load_febrl4(return_links=True)[1].to_csv("load_feb_dup.csv"))


hospital_accounts = pd.read_csv('duplicateDataset01.csv')
hospital_reimbursement = pd.read_csv('duplicatedataset02.csv')


indexer = recordlinkage.Index()
indexer.block('last_name')

candidates = indexer.index(hospital_accounts, hospital_reimbursement)
print(len(candidates))

compare = recordlinkage.Compare()
compare.string('last_name',
            'last_name',
            threshold=0.65,
            label='FNAME')
compare.string('Address2',
            'Address2',
            method='jarowinkler',
            threshold=0.65,
            label='ADDRESS')



features = compare.compute(candidates, hospital_accounts,
                        hospital_reimbursement)


features.to_csv("test_record.csv",index=True)