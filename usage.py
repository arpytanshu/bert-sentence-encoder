#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 01:17:07 2020

@author: ansh
"""

from BertEncoder import BertSentenceEncoder


sentences = ['The black cat is lying dead on the porch.',
             'The way natural language is interpreted by machines is mysterious.',
             'Fox jumped over dog.']


BE = BertSentenceEncoder(model_name='bert-base-cased')





# Encode sentences to get embeddings for each word withot pooling
# specify from which layer to get the embeddings in layer parameter
word_encodings = BE.encoder(sentences, layer = -2, pooling_method = None)

'''
>>> [print(x.shape) for x in word_encodings]

torch.Size([1, 12, 768])
torch.Size([1, 13, 768])
torch.Size([1, 7, 768])
'''





# Encode sentences to get a fixed dimension embedding for each sentence,
# which is pooled along along all words using one of the pooling methods
sentence_encodings = BE.encoder(sentences, layer = -2, pooling_method = 'mean')

'''
>>> [print(x.shape) for x in sentence_encodings]

torch.Size([768])
torch.Size([768])
torch.Size([768])
'''


