import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np



class CaseClassifier(nn.Module):

    def __init__(self, config): 
        super(CaseClassifier, self).__init__()
        #Instantiating BERT model object 
        # print(config["model_type"])
        # print("hello") 
        # print(config)
        self.config = config
       
        self.bert_layer = AutoModel.from_pretrained(config["model_type"])
        

        self.cls_layer = nn.Linear(config["embedding_size"], 1) 

        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.drop_layer = nn.Dropout(p=config["dropout"])

    def forward(self, seq, attn_masks): 
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        outputs = self.bert_layer(seq, attention_mask = attn_masks, output_hidden_states = True)

        #Obtaining the representation of [CLS] head (the first token)
        # cls_rep = outputs.last_hidden_state[:, 0]
        if seq.is_cuda:
            dummy_tensor = torch.tensor([0.0]).cuda(seq.device)
        else: 
            dummy_tensor = torch.tensor([0.0])


        hids = outputs.last_hidden_state  # it is (batch_size, sequence_length, hidden_size)
        s = hids.size()
        # print(s)
        # print(attn_masks.size())
        mask = attn_masks.unsqueeze(-1)
        mask = mask.repeat([1, 1, s[2]])
        # print(mask.size())
        # mask = torch.reshape(mask, [s[0],s[1]*s[2]])
        hids = torch.where(mask==1, hids, dummy_tensor)
        
        # print(s)
        # only use cls embedding 
        hid_rep = hids[:, 0] 
        

        # print("cls_rep: ", len(cls_rep))
        # cls_rep = torch.cat((hids[-1][:, 0], hids[-2][:, 0], hids[-3][:, 0]), -1)

        # cls_rep = torch.cat((cls_rep.float(), publication_types_index.float()), 1)
        # concate cls with publication_type index 

        # add the dropout 
        hid_rep = self.drop_layer(hid_rep)

        #Feeding cls_rep to the classifier layer
        logits = self.cls_layer(hid_rep)

        logits = self.sigmoid(logits) 

        return logits

