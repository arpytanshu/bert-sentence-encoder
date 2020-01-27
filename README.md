# bert-sentence-encoder
Encode sentences to fix length vectors using pre-trained bert from huggingface-transformers


## Usage
```
from BertEncoder import BertSentenceEncoder
BE = BertSentenceEncoder(model_name='bert-base-cased')

sentences = ['The black cat is lying dead on the porch.',
             'The way natural language is interpreted by machines is mysterious.',
             'Fox jumped over dog.']
  
  
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
# which is pooled along along all words using one of the pooling methods ['max', 'mean' & 'max-mean']

sentence_encodings = BE.encoder(sentences, layer = -2, pooling_method = 'mean')
'''
>>> [print(x.shape) for x in sentence_encodings]
torch.Size([768])
torch.Size([768])
torch.Size([768])
'''
```

## Evaluation
A fixed length vector representation for each sentence is obtained if pooling is enabled.
To get a sense if the sentence vectors make sense, we evaluate the embeddings using pairs of duplicate sentences and pairs of different sentence.

### Evaluation Data
These sentence samples were obtained from the quora-question-pairs dataset from kaggle.

- **Example of Duplicate Sentence pairs:**  
  - How can I add photos or video on Quora when I want to answer?  
    How do I add a photo to a Quora answer?  
  
  - What are some of the most mind-blowing facts about Bengaluru?  
    What are some interesting facts about Bengaluru?  
    
  - Is a mental illness a choice? Does someone decide to have one or not?  
    Is mental illness is a choice?  
  
- **Example of different sentence pairs:**  
  
  - What is the best brand in power banks for smartphones?  
    What are some best power banks?  
   
  - What is it like to work with an executive recruiter?  
    What is the work of an executive recruiter like?  
 
  - Have you ever met an upcoming actor, actress or singer who you knew would go far in their career?  
    What is the best performance ever by a leading actor/actress in a TV series? Why?  
    
As one may notice, the sentences that are not duplicate, also share a lot of common words with their pairs. However the semantic meaning of the sentence is different.  

### Evaluation Method

We use L1 & L2 distance and Cosine similarity between the vector representation of pairs of words.
The distance between duplicate sentence pairs were always lower when compared to distance between different sentence pairs.
And the Similarity was higher for similar pairs of sentences.  
  
We used a sample of 200 pairs each of similar and different sentences, and got the sentence embeddings for all sentences using BertSentenceEncoder and pooled along all the words to get a fixed size vector. These vectors were used to calculate distance / similarity with their pairs and then meaned across all samples. We evaluated on embeddings from different layers of Bert.  
  
Since BERT is a model pretrained with a bi-partite target: masked language model and next sentence prediction. The last layer is trained in the way to fit this target, making it too “biased” to those two targets. For the sake of generalization, we could simply take the second-to-last layer and do the pooling.  


### Results

Layer = -1 is the last layer  
Layer = -2 is the second-to-last layer and so on  

- Max Pooling across words  

![](https://github.com/arpytanshu/bert-sentence-encoder/blob/master/graphs/max_L2_200.png) ![](https://github.com/arpytanshu/bert-sentence-encoder/blob/master/graphs/max_L1_200.png) ![](https://github.com/arpytanshu/bert-sentence-encoder/blob/master/graphs/max_CS_200.png)


- Mean Pooling across words  

![](https://github.com/arpytanshu/bert-sentence-encoder/blob/master/graphs/mean_L2_200.png) ![](https://github.com/arpytanshu/bert-sentence-encoder/blob/master/graphs/mean_L1_200.png) ![](https://github.com/arpytanshu/bert-sentence-encoder/blob/master/graphs/mean_CS_200.png)
