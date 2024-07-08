import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, BertModel
import torch.nn as nn


class CustomerBert(nn.Module):
    def __init__(self, name_or_model_path = "bert-base-uncased", n_classes = 2):
        super(CustomerBert, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(name_or_model_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes) 

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids = input_ids, attention_mask = attention_mask)
        x = self.classifier(x.pooler_output)

        return x

model = CustomerBert()
#model.load_state_dict(torch.load("my_custom_bert.pth")) #load the model

def classfier_fn(text:str):
    labels = ["negative","positive"]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, padding = "max_length", max_length = 250,
        truncation = True,  return_tensors = "pt",
        )
    
    output = model(input_ids= inputs["input_ids"], attention_mask = inputs["attention_mask"])
    _, pred = output.max(1)
    
    return labels[pred.item()]
    # print(pred.item())


# if __name__ == "__main__":
#     text = "Hello world"
#     print(classfier_fn(text=text))
#     return labels(pred.item())

demo = gr.Interface( 
    fn=classfier_fn, 
    inputs=["text"],
    outputs = ["text"],
    )

demo.launch()