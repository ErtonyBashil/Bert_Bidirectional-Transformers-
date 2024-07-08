
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertModel
from sklearn.model_selection import train_test_split
#from tensorflow.keras.optimizers import Adam
#from keras.optimizers import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
from torch.optim import Adam

class IMDBDataset(Dataset):
    def __init__(self, device, csv_file, name_or_model_path = "bert-base-uncased", max_length=250):

        self.device = device
        self.df = pd.read_csv(csv_file)
        self.labels = self.df.toxic.unique()
        # labels_dict = dict()
        # for idx, l in enumerate(self.labels):
        #     labels_dict[l] = idx

        # self.df["toxic"] = self.df["toxic"].map(labels_dict)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_model_path)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        review_text = self.df.comment_text[idx]
        label_review = self.df.toxic[idx]

        inputs = self.tokenizer(review_text, padding = "max_length", max_length = self.max_length,
        truncation = True,  return_tensors = "pt",
        )
        labels = torch.tensor(label_review)

        return{
            "inputs": inputs["input_ids"].squeeze(0).to(self.device),
            "attention_mask": inputs["attention_mask"].squeeze(0).to(self.device),
            "Labels": labels.to(self.device)
        }


class CustomerBert(nn.Module):

    def __init__(self, name_or_model_path = "bert-base-uncased", n_classes = 2): # model du bert a choisir
        super(CustomerBert, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(name_or_model_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes) # ca va nous permettre de prendre la sorite du model
                                        # on fait une projection lineaire et faire la classification binaire

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids = input_ids, attention_mask = attention_mask)
        x = self.classifier(x.pooler_output)

        return x

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)


def training_step(model, data_loader, loss_fn, optimizer):
    model.train()

    total_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        input_ids = data["inputs"]
        attention_mask = data["attention_mask"]
        labels = data["Labels"]

        optimizer.zero_grad()

        output = model(input_ids=input_ids, attention_mask = attention_mask)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    return total_loss / len(data_loader.dataset)



def evaluation(model, test_dataloader, loss_fn):
    model.eval()

    correct_predictions = 0
    losses = []

    for data in tqdm(test_dataloader, total=len(test_dataloader)):

        input_ids = data["inputs"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]

        output = model(input_ids=input_ids, attention_mask = attention_mask)
        _, pred = output.max(1)

        correct_predictions += torch.sum(pred == labels)
        loss = loss_fn(output, labels)
        losses.append(loss.item())

    return np.mean(losses), correct_predictions / len(test_dataloader.dataset)


def main():
    print("Training...")

    N_EPOCHS = 8
    LR = 2e-5
    BATCH_SIZE = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = IMDBDataset(csv_file="train.csv", device = device, max_length = 100)
    test_dataset = IMDBDataset(csv_file="test.csv", device = device, max_length = 100)

    train_dataloader = DataLoader(dataset = train_dataset, batch_size= BATCH_SIZE)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size= BATCH_SIZE)

    model = CustomerBert()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    #optimizer = optim.SGD(model.parameters(), lr=LR)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        loss_train = training_step(model, train_dataloader, loss_fn, optimizer)
        loss_eval, accuracy = evaluation(model, test_dataloader, loss_fn)

        print(f"Train Loss :{loss_train} | Eval loss: {loss_eval} | Accurary: {accuracy}")

    #save the model
    torch.save(model.state_dict(), "my_custom_bert.pth")


if __name__ == "__main__":
    main()
#     model = CustomerBert()
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     text = "Hello, my dog is cute. I am a student"
#     inputs = tokenizer(text, return_tensors = "pt")
#     last_hidden_state = model(input_ids = inputs["input_ids"])
#     print(last_hidden_state)