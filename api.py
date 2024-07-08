from fastapi import FastAPI
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, BertModel
import torch.nn as nn
from pydantic import BaseModel

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RequestPost(BaseModel):
    text:str


class CustomerBert(nn.Module):
    def __init__(self, name_or_model_path = "bert-base-uncased", n_classes = 2):
        super(CustomerBert, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(name_or_model_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes) 

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids = input_ids, attention_mask = attention_mask)
        x = self.classifier(x.pooler_outtput)

        return x

model = CustomerBert()
model.load_state_dict(torch.load("my_custom_bert.pth")) #load the model


def classfier_fn(text:str):
    labels = {0: "Negative", 1: "postive"}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = self.tokenizer(text, padding = "max_length", max_length = self.max_length,
        truncation = True,  return_tensors = "pt",
        )
    
    output = model(input_ids= inputs["inputs_ids"], attention_mask = inputs["attention_mask"])
    _, pred = output.max(1)

    return labels(pred.item())


@app.get("/")
def read_root():
    return {"Hello":"World" }

@app.get("/predict")
def prediction(request:RequestPost):
    return {
        "prediction": classfier_fn(request.text)
        }

