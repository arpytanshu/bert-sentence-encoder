#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 01:17:07 2020

@author: ansh
"""

import numpy as np
import pandas as pd


def get_evaluation_data(num_samples):
    # Get in data for evaluation
    similar_sentences = pd.read_csv('./evaluation_data/duplicate.csv', nrows=num_samples)
    different_sentences = pd.read_csv('./evaluation_data/different.csv', nrows=num_samples)
    
    similar_q1 = list(similar_sentences.question1.values)
    similar_q2 = list(similar_sentences.question2.values)
    
    different_q1 = list(different_sentences.question1.values)
    different_q2 = list(different_sentences.question2.values)
    return similar_q1, similar_q2, different_q1, different_q2




def l2_distance(s1_bert_embedding, s2_bert_embedding, means=True):
    '''
    inputs are expected to be in shape [num_sentences x embedding_dimension]
    '''
    eud = np.linalg.norm(s1_bert_embedding - s2_bert_embedding, axis=1)
    if means:
        return eud.mean()
    else:
        return eud


def l1_distance(s1_bert_embedding, s2_bert_embedding, means = True):
    '''
    inputs are expected to be in shape [num_sentences x embedding_dimension]
    '''
    l1 = np.sum(np.abs(s1_bert_embedding - s2_bert_embedding), axis=1)
    if means:
        return l1.mean()
    else:
        return l1


def cos_sim(s1_bert_embedding, s2_bert_embedding, means = True):
    '''
    inputs are expected to be in shape [num_sentences x embedding_dimension]
    '''
    num = np.diag(np.matmul(s1_bert_embedding, s2_bert_embedding.T))
    den = ( np.linalg.norm(s1_bert_embedding, axis=1) * np.linalg.norm(s2_bert_embedding, axis=1) )
    cos_s = num / den
    if means:
        return cos_s.mean()
    else:
        return cos_s
