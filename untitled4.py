# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:58:27 2023

@author: onais
"""

from transformers import AutoTokenizer, XLMRobertaXLForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-xlarge")
model = XLMRobertaXLForTokenClassification.from_pretrained("xlm-roberta-xlarge")

inputs = tokenizer(
    "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_token_class_ids = logits.argmax(-1)

# Note that tokens are classified rather then input words which means that
# there might be more predicted token classes than words.
# Multiple token classes might account for the same word
predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

labels = predicted_token_class_ids
loss = model(**inputs, labels=labels).loss