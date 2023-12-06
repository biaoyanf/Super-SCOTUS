import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import json
import numpy as np
import os
import pickle

class CaseDataset(Dataset): 
    
    # def __init__(self, model_type, label_path, evidence_path, maxlen): 
    def __init__(self, config, data_path):
        
        # self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_type"]) 

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id 
        print("cls_id, sep_id, pad_id: ", cls_id, sep_id, pad_id)


        self.maxlen = config["max_token"]
        self.config = config
        
        self.ids = []

        #get the favor side 
        self.labels = []

        # construct the dataset   each contains  

        # - case ID  -> string  -  to map back 
        # - sentences/utterances -> list   
        # - speakers  -> list - help to separate the utterances later    
        # - side -> so it is helpful to select the side too? 
        # - winning side -> string/numeric  - it could be not clear too right? 

        self.text = []

        label_count = {0: 0, 1: 0}

        selected_cases = None 
        with open(data_path, "r") as fr:  
            selected_cases = json.load(fr) 
        assert selected_cases

        for case in selected_cases: 
            
            # process the label first  
            
            # ------------ win_side  --------------
            # if case["win_side"] not in [0,1]: continue  # control the situation where the case has no output label
            
            # assert  case["win_side"] in [0, 1] 
            
            # label = case["win_side"]

            # if case["win_side"] not in [0,1]: continue  # control the situation where the case has no output label 
            # topic 
            # label = case["scdb_elements"]["issueArea"]
            # if np.isnan(label):  
            #     count+=1 
            #     continue   # mean that it is nan and we do not use this case 
            # conservative/liberal 
            # label = case["scdb_elements"]["lcDispositionDirection"] 

            # ------------ direction  --------------
            
            label = case["scdb_elements"]["decisionDirection"]
            assert label in [1, 2] 
            
            if label == 2: label = 0

            self.ids.append(case["id"]) 
            # self.win_side.append(case["win_side"]) 
            self.labels.append(label)
            label_count[label]+=1

            # now processing the text  

            # first get the text information   

            current_text = None 
            if self.config["input_type"] == "Conversation": 
                # print("using Conversation text... ")
                text = []
                for utt in case["convos"]["utterances"]: 
                    for u in utt:
                        # text.append(u["text"])

                        # here to control the text side  

                        # if  using the full content 
                        if self.config["get_justices_text"] and self.config["get_non_justices_text"]: 
                            text.append(u["text"])

                        # using only justices  
                        elif self.config["get_justices_text"] and not self.config["get_non_justices_text"]: 
                            if case["convos"]["speaker"][u["speaker_id"]]["type"] == "J":
                                text.append(u["text"]) 
                                
                        # using only non justices  
                        elif not self.config["get_justices_text"] and self.config["get_non_justices_text"]: 
                            if case["convos"]["speaker"][u["speaker_id"]]["type"] != "J":
                                text.append(u["text"])
                            
                    # text.append(utt["text"])
                # self.text.append(text)  
                current_text = " ".join(text) 

            elif self.config["input_type"] == "Syllabus": 
                # print("using Syllabus text. ")
                # self.text.append([case["justia_sections"]["Syllabus"]])  # input as a list, not a string, to fit the conversation 

                current_text = case["justia_sections"]["Syllabus"] 

            # elif self.config["input_type"] == "Opinion":  
            #     # print("using Opinion text.") 

            #     assert len(case["justia_sections"].keys()) >1 
            #     assert list(case["justia_sections"].keys())[0] == "Syllabus"
            #     self.text.append([case["justia_sections"][list(case["justia_sections"].keys())[1]]])  # input as a list, not a string, to fit the conversation 



            # now we tokenizer the current text 
            assert current_text 
            
            if config["lower_case"]:  current_text = current_text.lower() 


            current_token_ids = self.tokenizer.encode(current_text, 
                                                        add_special_tokens=False, # Add special tokens for BERT
                                                        )  

            # start tokenizinig 

            if not self.config['last_text']:  # using the first max tokens 
                current_token_ids = current_token_ids[: self.maxlen-2]  # cuz we need to add the cls and sep 
            else: 
                current_token_ids = current_token_ids[-(self.maxlen-2): ]  # cuz we need to add the cls and sep 
                            
            current_token_ids = [cls_id]+current_token_ids+[sep_id] 

            
            len_current_token_ids = len(current_token_ids)
            if len_current_token_ids< self.maxlen: 
                current_token_ids = current_token_ids + [pad_id]*(self.maxlen-len_current_token_ids) 

            self.text.append(current_token_ids) 

        print(label_count, sum(label_count.values()))



    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index): 
        ids = self.ids[index]
        # ids = torch.tensor(ids)
 
        label = self.labels[index]
        label = torch.tensor(label)
        
        
        current_token_ids = self.text[index] 
        
        # print("current_token_id: ", current_token_ids)
       
        # tokens_ids = self.tokenizer.convert_tokens_to_ids(current_token) #Obtaining the indices of the tokens in the BERT Vocabulary

        
        tokens_ids_tensor = torch.tensor(current_token_ids) #Converting the list to a pytorch tensor
        # print("tokens_ids_tensor", tokens_ids_tensor.size())
        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()
        # print(label)
        # print("label:", label)
        # print("publication_types_index:", publication_types_index)
        # print(tokens_ids_tensor.shape) 
        # print(attn_mask.shape)
        # print(label)
        # # print(ids) 
        # print()
        return tokens_ids_tensor, attn_mask, label, ids
    

