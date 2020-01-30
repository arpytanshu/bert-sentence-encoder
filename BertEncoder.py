#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 01:17:07 2020

@author: ansh
"""

     
import sys
import time
import torch
import numpy as np
from tqdm import tqdm, trange
from transformers import BertModel, BertTokenizer, BertConfig


class BertSentenceEncoder():
    def __init__(self, model_name='bert-base-cased'):
        '''
        Parameters
        ----------
        model_name : string, optional
            DESCRIPTION. The default is 'bert-base-cased'.
            
            Find a list of usable pre-trained bert models from:
                https://huggingface.co/transformers/pretrained_models.html
        '''

        self.model_name =   model_name
        self.config =       BertConfig.from_pretrained(self.model_name, output_hidden_states=True, training=False)
        self.model =        BertModel.from_pretrained(self.model_name, config=self.config)
        self.tokenizer =    BertTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.pooling_methods = ['max', 'mean', 'max-mean']
        self.model.eval()
    
    def __repr__(self):
        return 'BertSentenceEncoder model:{}'.format(self.model_name)
    
    def _mean_pooler(self, encoding):
        return encoding.mean(dim=1)
    
    def _max_pooler(self, encoding):
        return encoding.max(dim=1).values
    
    def _max_mean_pooler(self, encoding):
        return torch.cat((self._max_pooler(encoding), self._mean_pooler(encoding)), dim=1)
    
    def _pooler(self, encodings, pooling_method):
        '''
        Pools the encodings along the time/sequence axis according
        to one of the pooling method:
            - 'max'      :  max value along the sequence/time dimension
                            returns a (batch_size x hidden_size) shaped tensor
            - 'mean'     :  mean of the values along the sequence/time dimension
                            returns a (batch_size x hidden_size) shaped tensor
            - 'max-mean' :  max and mean values along the sequence/time dimension appended
                            returns a (batch_size x 2*hidden_size) shaped tensor
                            [ max : mean ]
        Parameters
        ----------
        encoding : list of tensor to pool along the sequence/time dimension.
        
        pooling_method : one of 'max', 'mean' or 'max-mean'
        
        Returns
        -------
        tensor of shape (batch_size x hidden_size).
        '''
        
        assert (pooling_method in self.pooling_methods), \
            "pooling methods needs to be one of 'max', 'mean' or 'max-mean'"
            
        if pooling_method   == 'max':       pool_fn = self._max_pooler
        elif pooling_method == 'mean':      pool_fn = self._mean_pooler
        elif pooling_method == 'max-mean':  pool_fn = self._max_mean_pooler
        
        pooled = pool_fn(encodings)
        
        return pooled
    

    
    def encoder(self, sentences, layer=-2, pooling_method = None, max_length=40 ):
     
        assert isinstance(sentences, list), \
            "parameter 'sentences' is supposed to be a list of string/s"
        assert all(isinstance(x, str) for x in sentences), \
            "parameter 'sentences' must contain strings only"
        
        '''
        model(input_tokens) returns a tuple of 3 elements.
        out[0] : last_hidden_state  of shape [ B x T x D ]
        out[1] : pooler_output      of shape [ B x D ]
        out[2] : hidden_states      13 tuples, one for each hidden layer
                                    each tuple of shape [ B x T x D ]        
        '''
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', max_length=max_length)['input_ids']
            encoded = self.model(input_ids)
                    
        if pooling_method in self.pooling_methods:
            pooled = self._pooler(encoded[2][layer], pooling_method)
            return pooled
        
        return encoded


def get_BE_batched(sentences, batch_size, BE=None):
    assert(BE), "Provide a BertSentenceEncoder object."
    l = len(sentences)
    embeddings = np.empty((0,768))    
    num_batches = int(l/batch_size) if l%batch_size==0 else int(l/batch_size)+1
    
    t = trange(num_batches, desc='Batch', leave=True)

    for i in t:
        # get start and end index for this batch
        if( i != int(l/batch_size) ):
            start   = (i*batch_size)
            end     = (i*batch_size)+batch_size   
        else:
            start   = int(l/batch_size)*batch_size
            end     = l
        t.set_description('Embedding batch => {} : {}'.format(start, end))
    
        # s = time.time()
        batch_embeddings = BE.encoder(sentences[start:end], layer = -2, pooling_method='mean')
        # e = time.time()    
        # print("Time elapsed: {} seconds.".format(e-s), file=sys.stderr)
        
        embeddings = np.append(embeddings, batch_embeddings, axis=0)
        
    return embeddings
    
