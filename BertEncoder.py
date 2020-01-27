#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 01:17:07 2020

@author: ansh
"""

     
import torch
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
        self.tokenizer =    BertTokenizer.from_pretrained(self.model_name)
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
            
        pooled = []
        if pooling_method == 'max':         pool_fn = self._max_pooler
        elif pooling_method == 'mean':      pool_fn = self._mean_pooler
        elif pooling_method == 'max-mean':  pool_fn = self._max_mean_pooler
        
        for encoded in encodings :       
            pooled.append(pool_fn(encoded).squeeze_())
        
        return torch.stack(pooled)
    
    def encoder(self, sentences, layer=-2, pooling_method = None ):
        '''
        Get the BERT embeddings for the sentence/s from
        the hidden layer specified in the layer parameter.
        Parameters
        ----------
        sentence : list of string
            list of string to be encoded.
            length of list is the batch_size
            
        layer : int
            the layer from which to get the encoding.
            -1 = last layer
            -2 = second last layer
            ...
            default = -2
            BERT is a model pretrained with a bi-partite target:
                masked language model and next sentence prediction.
            The last layer is trained in the way to fit this target,
            making it too “biased” to those two targets. For the sake
            of generalization, we could simply take the second-to-last layer
            and do the pooling.
            
        pooling_method : one of 'max', 'mean' or 'max-mean'
            if None, returns the word embeddings without any pooling.
            
        Returns
        -------
        if pooling_method is None, returns a list of tensors of shape
        (sequence_length x hidden_size). [one tensor for each sentence]
        which are the word embeddings for the sentence without pooling.
        
        if pooling_method is specified, returns a list of tensor,
        one tensor of shape (hidden_size) for each sentence.
        
        '''
        
        assert isinstance(sentences, list), \
            "parameter 'sentences' is supposed to be a list of string/s"
        assert all(isinstance(x, str) for x in sentences), \
            "parameter 'sentences' must contain strings only"
        
        encodings = []
        with torch.no_grad():
            for sentence in sentences:
                input_id = self.tokenizer.encode(sentence, return_tensors='pt')
                encoded = self.model(input_id)
                encodings.append(encoded[2][layer])
            
        if pooling_method in self.pooling_methods:
            pooled = self._pooler(encodings, pooling_method)
            return pooled
        
        return encodings
