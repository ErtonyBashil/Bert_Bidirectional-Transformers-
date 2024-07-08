# BERT: Pre-training of Deep Bidirectional Transformers for  Language Understanding
BERT: Pre-training of Deep Bidirectional Transformers for  Language Understanding

BERT stands for Bidirectional Encoder Representations from Transformers designed to pre-train deep bidirectional representations from unlabeled 
text by jointly conditioning on both left and right context in all layers. BERT alleviates the previously mentioned unidi rectionality constraint by using a “masked lan guage model” (MLM) 

There are two steps in our framework: pre-training and fine-tuning.
During pre-training, the model is trained on unlabeled data over different pre-training tasks. 
For fine tuning, the BERT model is first initialized with the pre-trained parameters, 
and all of the param eters are fine-tuned using labeled data from the downstream tasks. 
Each downstream task has sep arate fine-tuned models, 
even though they are ini tialized with the same pre-trained parameters

### 1. Model Architecture 

BERT’s model architec ture is a multi-layer bidirectional Transformer en coder based on the original 
implementation de scribed in Vaswani et al. (2017) and released in the tensor2tensor library
