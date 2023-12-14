#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import os
import logging
import numpy as np
import random
from tqdm import tqdm
import time
import pandas as pd

from transformers import LongformerModel, LongformerForSequenceClassification, LongformerForMultipleChoice
from transformers import AutoTokenizer, AutoModel, RobertaModel, RobertaTokenizer, RobertaConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from focal_loss.focal_loss import FocalLoss

logging.basicConfig(filename=f'./logs/train_{time.asctime().replace(" ","_")}.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a stream handler to print log messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

torch.manual_seed(40)
np.random.seed(40)
random.seed(40)
torch.cuda.manual_seed(40)
torch.backends.cudnn.deterministic = True


# In[2]:


# Define the path to the CSV file
train_csv_file = "/data1/debajyoti/colie/train.csv"
val_csv_file = "/data1/debajyoti/colie/valid.csv"

# Read the CSV file
train_labels = pd.read_csv(train_csv_file)
val_labels = pd.read_csv(val_csv_file)
val_labels


# In[3]:


train_labels.BOOK_id[0]


# In[4]:


# Define the path to the train folder
train_folder = "/data1/debajyoti/colie/train/train/"
# Define the path to the validation folder
val_folder = "/data1/debajyoti/colie/valid/valid/"



def create_df(folder, label):
    # Initialize empty lists to store the data
    text_data = []
    labels = []
    for index in label.index:
        # filename = df_labels.BOOK_id[index]
        # print(filename)
        # print(df_labels['BOOK_id'][index], df_labels['Epoch'][index])
        file_name = label['BOOK_id'][index]  # Assuming 'File Name' is the column name for the file names in the CSV

        # Construct the file path
        file_path = os.path.join(folder, file_name)

        # Read the text from the file
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            text = file.read()

        # Append the text and label to the respective lists
        text_data.append(text)
        labels.append(label['Epoch'][index].strip())  # Assuming 'Label' is the column name for the labels in the CSV
        # break
    return text_data, labels

train_data, train_label = create_df(train_folder, train_labels)
val_data, val_label = create_df(val_folder, val_labels)

# Create a dataframe from the lists
train = pd.DataFrame({'text': train_data, 'label': train_label})
val = pd.DataFrame({'text': val_data, 'label': val_label})
print(train.head(), val.head())
print(train.shape, val.shape)


# In[5]:


label_dic = {'Romanticism':0,
            'Viktorian':1,
            'Modernism':2,
            'PostModernism':3,
            'OurDays':4}
train['label'] = train['label'].map(label_dic)
val['label'] = val['label'].map(label_dic)


# In[6]:


# Length of text
def length (txt):
    length = len(txt.split())
    return length

txt_length = train['text'].apply(lambda x: length(x))
print(txt_length.sort_values(ascending = False))


# In[7]:


val['label'].value_counts()


# In[8]:


# model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
# tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')


# In[9]:


max_length= 512
class CustomDataset(Dataset):
    def __init__(self, tokenizer, df):
        # Initialize thetokenizer
        self.tokenizer = tokenizer

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Get the text and label from the dataframe
        text = self.df.iloc[index]['text']
        label = self.df.iloc[index]['label']

        # Tokenize the text and convert it to input IDs
        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens=False,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )


        # Return the input IDs and label as PyTorch tensors
        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            # 'token_type_ids': inputs['token_type_ids'][0],
            'label': torch.tensor(label, dtype=torch.int64),
        }

# datasetclass = CustomDataset(tokenizer, train)
train_dataset = CustomDataset(tokenizer, train)
val_dataset = CustomDataset(tokenizer, val)

# DataLoader
batch_size = 8
train_dataloader = tqdm(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
val_dataloader = tqdm(DataLoader(val_dataset, batch_size=batch_size, shuffle=True))


# In[ ]:


class TransformerModel(nn.Module):
    def __init__(self, num_labels):
        super(TransformerModel, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained('roberta-large')
        # self.xlnet.resize_token_embeddings(num_tokens)
        # self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads), num_layers=num_layers)
        #self.transformer_decoder = TransformerDecoder(TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads), num_layers=num_layers)
        #self.transformer = Transformer(nhead=16, num_encoder_layers=6, num_decoder_layers = 6)
        self.decoder = nn.Linear(self.roberta.config.hidden_size, num_labels) 
        # self.fc1 = nn.Linear(num_tokens, 2)
        # self.fc2 = nn.Linear(num_tokens, 2)
        # self.fc3 = nn.Linear(num_tokens, 5)
        # self.num_classes = num_classes
        # self.classifiers = nn.ModuleList([nn.Linear(self.roberta.config.hidden_size, num_classes[i]) for i in range(len(num_classes))])
        # self.classifiers = nn.ModuleList([nn.Linear(num_tokens, num_classes[i]) for i in range(len(num_classes))])
        # self.tanh = nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):  # src = [bsz, seq_len]
        roberta_output = self.roberta(input_ids=input_ids).pooler_output
        # print(long_output.shape)
        # roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state = outputs.last_hidden_state # Shape: (batch_size, sequence_len, hidden_size)
        # src_embedded = last_hidden_state
        # src_embedded = self.roberta.embeddings(src) # Use RoBERTa model to embed source sequence output: [bsz, seq_len, features,i.e. hidden_dim] [20, 100, 768]
        # print("shape of roberta embeddings:", src_embedded.shape)
        #tgt_embedded = self.roberta.embeddings(tgt) # Use RoBERTa model to embed target sequence
        # src_embedded = src_embedded # output: [bsz, seq_len, features] 
        # src_embedded = torch.cat([t1,t2,t3, src_embedded],1)

        # t1 = torch.cat(src_embedded.size(0) * [t1])
        # t2 = torch.cat(src_embedded.size(0) * [t2])
        # t3 = torch.cat(src_embedded.size(0) * [t3])
        # t = torch.stack([t1,t2,t3], dim=1)
        # task_embedded = torch.cat([t, src_embedded],1)  # output shape: [bsz, seq_len, features] [8, 203, 768]

        # memory = self.transformer_encoder(src_embedded)  # output shape: [bsz, seq_len, features] [8, 203, 768]
        # print("shape after transformer encoder layer:", memory.shape)
        #output = self.transformer_decoder(tgt_embedded, memory)
        #print("shape after transformer decoder layer:", output.shape)

        output = self.decoder(roberta_output)  # output shape: [bsz, seq_len, vocab_size] [8, 203, 50k]
        # print("shape after transformer decoder layer:", output.shape, output.dtype)
        # task1_output = self.fc1(output[:,0,:])
        # task2_output = self.fc2(output[:,1,:])
        # task3_output = self.fc3(output[:,2,:num_classes])
        # ae_output = output[:,len(self.num_classes):,:]
        # ae_output = output[:,:,:]
        # print("shape after final linear layer:", output.shape)
        # task_logits = [classifier(pooled_output) for classifier in self.classifiers]
        # task_logits = []

        # pooled_outputs = [output[:,i,:] for i in range(len(self.num_classes))] # output shape : [bsz, 1, vocab_size]

        # for classifier, pooled_output in zip(self.classifiers, pooled_outputs):
        #     # pooled_output = self.tanh(pooled_output)
        #     logits = classifier(pooled_output)
        #     task_logits.append(logits)

        output = self.softmax(output)
        
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


num_labels = 5

model = TransformerModel(num_labels).to(device)


# In[ ]:


# num_epochs = 5
learning_rate = 2e-6
class_weights = torch.tensor([0.35, 0.03, 0.03, 0.25, 0.34]).to(device)

# with weights 
# The weights parameter is similar to the alpha value mentioned in the paper
criterion = FocalLoss(gamma=0.7, weights=class_weights).to(device)


# Set optimizer and learning rate scheduler
# criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


# In[ ]:


def get_labels(logit, targets):
    """
    Calculate accuracy and macro F1 score for each class
    """
    # pos = list(task_dict.keys()).index(task_name)
    # mask = torch.arange(targets.shape[0]).to(device)
    # task_idx = mask[targets[:,pos] != 99]
    output = logit
    true_label = targets
    # print("shapes for label:", output.shape, true_label.shape)
    pred_label = torch.argmax(output, dim=1).flatten().tolist()
    true_label = true_label.flatten().tolist()


    return pred_label, true_label


# In[ ]:


current_train_loss = []

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 1
    start_time = time.time()
    num_batches = len(train_dataset) // batch_size
    for batch, i in enumerate(train_dataloader):
        data, mask, targets = i.values()
        data = data.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        # print(data.dtype)        
        # print(data.shape)
        # task_logits, ae_output = model(data)
        output = model(data, mask)
        # t1_out, t2_out, t3_out, auto_output = model(data, t1, t2, t3)
        # loss = custom_loss(logits_task1, logits_task2, logits_task3, targets)
        # print("shape:", data.shape, targets.flatten().shape)
        # print("datatype:", data.dtype, targets.flatten().dtype)
        loss = criterion(output, targets.flatten())


        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            # ppl = np.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.7f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.5f}')
            total_loss = 0
            start_time = time.time()
        
        if batch == 100:
            break
    current_train_loss.append(cur_loss)


# In[ ]:


def evaluate(model: nn.Module) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    # src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        predictions = []
        true_labels = []
        for batch, i in enumerate(val_dataloader):
            data, mask, targets = i.values()
            data = data.to(device)
            mask = mask.to(device)
            targets = targets.to(device)
            seq_len = data.size(1)
            # logits_task1, logits_task2, logits_task3, ae_output = model(data, mask)
            # task_logits, ae_output = model(data)
            # task_logits, ae_output = model(data, mask)
            output = model(data, mask)
            # t1_out, t2_out, t3_out, auto_output = model(data, t1, t2, t3)
            # loss = custom_loss(logits_task1, logits_task2, logits_task3, targets)
            # loss = custom_loss(logits_task1, logits_task2, logits_task3, ae_output, data, targets)
            loss = criterion(output, targets.flatten())

            total_loss += seq_len * loss.item()

            #get the labels for classification report
            pred_label, true_label = get_labels(output, targets)
            predictions.extend(pred_label)
            true_labels.extend(true_label)
            # if batch == 100:
            #     break

    # Compute overall classification report
    logging.info(f"\n Scores:")
    logging.info(f"\n {classification_report(true_labels, predictions)}")
    return total_loss / (len(val_dataset) - 1)


# In[ ]:


logging.info(f"#"* 89)
logging.info(f"\n DESCRIPTION-> \n logic: longformer + linear_layer + loss_reweighting(100 batches), model: {tokenizer.name_or_path}, lr:{learning_rate}, max_seq_length: {max_length}")
logging.info('#' * 89)


# In[ ]:


best_val_loss = float('inf')
current_val_loss = []   # for plotting graph of val_loss
epochs = 40
early_stop_thresh = 3

tempdir = '/data1/debajyoti/colie/.temp/'
best_model_params_path = os.path.join(tempdir, f"best_model_params_{time.asctime().replace(' ','_')}.pt")

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model)
    current_val_loss.append(val_loss)
    # val_ppl = np.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    logging.info('-' * 89)
    logging.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'valid loss {val_loss:5.5f}')
    logging.info('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_params_path)
    elif epoch - best_epoch > early_stop_thresh:
        logging.info("Early stopped training at epoch %d" % epoch)
        break  # terminate the training loop

    scheduler.step()
model.load_state_dict(torch.load(best_model_params_path)) # load best model states


# In[ ]:


(1/43)/((1/30) + (1/377) + (1/327) + (1/43) + (1/31))

