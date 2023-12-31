import time 

from torch.utils.data import DataLoader
from CaseDataset import CaseDataset, CaseDataset_hierarchy,CaseDataset_hierarchy_fixed
from CaseClassifier import CaseClassifier, CaseClassifier_hierarchy, CaseClassifier_hierarchy_fixed
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import pyhocon
import os
import errno

import json

from sklearn.metrics import f1_score,accuracy_score, classification_report

def initialize_from_env():
#   if "GPU" in os.environ:
#     set_gpus(int(os.environ["GPU"]))
#   else:
#     set_gpus()

  name = sys.argv[1]
  print("Running experiment: {}".format(name))

  config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]
  config["log_dir"] = mkdirs(os.path.join(config["log_root"], name))

  print(pyhocon.HOCONConverter.convert(config, "hocon"))
  return config

def mkdirs(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
  return path






def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    # soft_probs = (probs > 0.5).long()
    # print("logits ", logits)
    # print("lables: ", labels)
    pred_arg_max = torch.argmax(probs, dim=1) # now gfet a list [1, 2, 0, ...]  -> to get the max index
    pred_arg_max = torch.squeeze(pred_arg_max)
    # ground_arg_max = torch.argmax(labels, dim=-1)
    # print(pred_arg_max)
    # print(pred_arg_max.tolist(), labels.tolist())
    # print((pred_arg_max == labels).float().tolist())
    # acc = (pred_arg_max == labels).float().mean() 


    flat_pred_arg_max = torch .reshape(pred_arg_max, (-1,))
    flat_labels = torch.reshape(labels, (-1,))
    f1 = f1_score(flat_labels.cpu(), flat_pred_arg_max.cpu(), average = "micro")  
    macrof1 = classification_report(flat_labels.cpu(), flat_pred_arg_max.cpu(),output_dict=True)["macro avg"]["f1-score"]
    acc = accuracy_score(flat_labels.cpu(), flat_pred_arg_max.cpu())
    # if f1 == 1.0:
    # print("prediction: ", flat_pred_arg_max)
    # print("gold_label: ", flat_labels)
    # print("overlap: ", set(flat_labels)-set(flat_pred_arg_max), set(flat_pred_arg_max)-set(flat_labels))
    # print("\n")

    return macrof1, acc, f1

def evaluate(net, criterion, dataloader, gpu):
    net.eval()

    mean_loss = 0
   
    count = 0
    # pred_results = {}
    if gpu >= 0:
        golds = torch.Tensor([]).cuda(gpu)
        preds = torch.Tensor([]).cuda(gpu) 
    else: 
        golds = torch.Tensor([]).cpu()
        preds = torch.Tensor([]).cpu()
   
    
    with torch.no_grad():
        for (seq, attn_masks, labels, ids) in dataloader:
            if gpu >= 0:
                seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)
            else:
                seq, attn_masks, labels = seq.cpu(), attn_masks.cpu(), labels.cpu()
                
            logits = net(seq.long(), attn_masks) 
   
            mean_loss += criterion(logits.squeeze(-1), labels.long()).item() 

            count += 1

            golds = torch.cat([golds, labels], 0) 
            preds = torch.cat([preds, logits], 0) 
            
         
    macrof1, acc, f1 = get_accuracy_from_logits(preds, golds)  

    return macrof1, acc, f1,  mean_loss / count

    

    

def evaluate_file(net, criterion, dataloader, file_type, gpu):
    net.eval()

    mean_loss = 0
    count = 0
    pred_results = {}
    if gpu >= 0:
        golds = torch.Tensor([]).cuda(gpu)
        preds = torch.Tensor([]).cuda(gpu) 
    else: 
        golds = torch.Tensor([]).cpu()
        preds = torch.Tensor([]).cpu()
   
    
    with torch.no_grad():
        for (seq, attn_masks, labels, ids) in dataloader:
            if gpu >= 0:
                seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)
            else:
                seq, attn_masks, labels = seq.cpu(), attn_masks.cpu(), labels.cpu()
                
            logits = net(seq.long(), attn_masks) 
            mean_loss += criterion(logits.squeeze(-1), labels.long()).item() 
            # tmp_macrof1, tmp_acc, tmp_f1 = get_accuracy_from_logits(logits, labels)
            # mean_acc += tmp_acc
            # mean_f1 += tmp_f1
            # mean_macrof1 += tmp_macrof1
            count += 1

            golds = torch.cat([golds, labels], 0) 
            preds = torch.cat([preds, logits], 0) 
            
            store_prediciton(pred_results, ids, logits) 
    # print(pred_results)
    with open("./{}/pred_{}.txt".format(config["log_dir"], file_type), "w") as fw: 
        fw.write(json.dumps(pred_results))
    # print("preds: ", preds) 
    # print("golds: ", golds)
    macrof1, acc, f1 = get_accuracy_from_logits(preds, golds)  

    return macrof1, acc, f1,  mean_loss / count

    # return mean_macrof1 / count, mean_acc / count, mean_f1 / count,  mean_loss / count

def store_prediciton(pred_results, ids, logits):
    # print("logits", logits)

    probs = torch.sigmoid(logits.unsqueeze(-1))
    pred_arg_max = torch.argmax(probs, dim=1) # now gfet a list [1, 2, 0, ...]  -> to 
    pred_arg_max = torch.squeeze(pred_arg_max)
    flat_pred_arg_max = torch.reshape(pred_arg_max, (-1,))
    

    if pred_arg_max.size() == torch.Size([]):  # after the squeeze pred_arg_max become an int as the batch size become 1. so we need to make it as a list again 
        # print("pred_arg_max: ", pred_arg_max)
        # print("pred_arg_max.size(): ", pred_arg_max.size())
        pred_arg_max = [pred_arg_max.tolist()]  
    else:  
        pred_arg_max = pred_arg_max.tolist() 

    for i, p in zip(ids, pred_arg_max):
        assert i not in pred_results
        pred_results[i] = p 

def store_result(data_type, acc, f1, macrof1, loss): 
    with open("./{}/evaluation_result.txt".format(config["log_dir"]), "a") as fw:
        fw.write("{}: Acc: {:.4f}, F1: {:.4f}, Macro F1: {:.4f}, Loss: {} \n".format(data_type, acc, f1, macrof1, loss)) 


if __name__ == "__main__":

    
    config = initialize_from_env()

    name = sys.argv[1]  
    if "hierarchy" not in name:  
        train_set = CaseDataset(config = config, data_path = config["train_label_path"])
        dev_set = CaseDataset(config = config, data_path = config["dev_label_path"])
        test_set = CaseDataset(config = config, data_path = config["test_label_path"])
    else: 
        if "fixed" not in name: 
            train_set = CaseDataset_hierarchy(config = config, data_path = config["train_label_path"])
            dev_set = CaseDataset_hierarchy(config = config, data_path = config["dev_label_path"])
            test_set = CaseDataset_hierarchy(config = config, data_path = config["test_label_path"])
        else:
            train_set = CaseDataset_hierarchy_fixed(config = config, data_path = config["train_label_path"])
            dev_set = CaseDataset_hierarchy_fixed(config = config, data_path = config["dev_label_path"])
            test_set = CaseDataset_hierarchy_fixed(config = config, data_path = config["test_label_path"])

    # print(test_set.ids, test_set.favor_side)

    #Creating intsances of training and development dataloaders
    train_loader = DataLoader(train_set, batch_size = config["batch_size"], num_workers = 1)
    dev_loader = DataLoader(dev_set, batch_size = config["batch_size"], num_workers = 1)
    test_loader = DataLoader(test_set, batch_size = config["batch_size"], num_workers = 1)

    print("Done preprocessing train, development, and test data.")
    

    # gpu = 5  #gpu ID
    gpu = config["gpu"]

    

    if "hierarchy" not in name: 
        print("Creating the CaseClassifier, initialised with pretrained {}".format(config["model_type"])) 
        net = CaseClassifier(config) 
    else: 
        if "fixed" not in name: 
            print("Creating the CaseClassifier_hierarchy, initialised with pretrained {}".format(config["model_type"])) 
            net = CaseClassifier_hierarchy(config)  
        else: 
            print("Creating the CaseClassifier_hierarchy_fixed, initialised with pretrained {}".format(config["model_type"])) 
            net = CaseClassifier_hierarchy_fixed(config)  

    net.load_state_dict(torch.load('./{}/sstcls_best.dat'.format(config["log_dir"])))
    if gpu > -1:
        net.cuda(gpu) #Enable gpu support for the model
    else:
        net.cpu()
    # net.cpu()
    print("Done loading the CaseClassifier.")

    criterion = nn.CrossEntropyLoss()
    # criterion = RegLoss_fix()
    opti = optim.Adam(net.parameters(), lr = config["lr"])


    # dev_load_file = "answer_merged_dev.txt"
    # with open

    # train_acc, train_f1, train_loss = evaluate_file(net, criterion, train_loader, "train", gpu)
    train_macrof1, train_acc, train_f1, train_loss = evaluate(net, criterion, train_loader, gpu)
    print("Trainset: Accuracy: {}; f1_score: {}; Macro f1_score: {}; Loss: {}".format(train_acc, train_f1, train_macrof1, train_loss))
    store_result("Train", train_acc, train_f1, train_macrof1, train_loss) 

    # dev_acc, dev_f1, dev_loss = evaluate(net, criterion, dev_loader, gpu)
    dev_macrof1, dev_acc, dev_f1, dev_loss = evaluate_file(net, criterion, dev_loader, "dev", gpu)
    print("Devset: Accuracy: {}; f1_score: {}; Macro f1_score: {}; Loss: {}".format(dev_acc, dev_f1, dev_macrof1, dev_loss))
    store_result("Dev", dev_acc, dev_f1, dev_macrof1, dev_loss)

    # test_acc, test_f1, test_loss = evaluate(net, criterion, test_loader, gpu)
    test_macrof1, test_acc, test_f1, test_loss = evaluate_file(net, criterion, test_loader, "test", gpu)
    print("Testset: Accuracy: {}; f1_score: {}; Macro f1_score: {}; Loss: {}".format(test_acc, test_f1,test_macrof1, test_loss))
    store_result("Test", test_acc, test_f1, test_macrof1, test_loss)


    print("Macro F1 in train, dev, and test:  {:.4f} &  {:.4f} &  {:.4f}".format(train_macrof1, dev_macrof1, test_macrof1))
    with open("./{}/evaluation_result.txt".format(config["log_dir"]), "a") as fw:
        fw.write("Macro F1 in train, dev, and test:  {:.4f} &  {:.4f} &  {:.4f}".format(train_macrof1, dev_macrof1, test_macrof1))
    #fine-tune the model
    # train(net, config, criterion, opti, train_loader, dev_loader, num_epoch, gpu)


    # net.load_state_dict(torch.load('./{}/sstcls_best.dat'.format(config["log_dir"])))

    # print("Best performance in Dev: acc, f1, loss")
    # dev_acc, f1, dev_loss = evaluate(net, criterion, dev_loader, gpu)
    # print(dev_acc, f1, dev_loss)

    # print("corresponding result in Train: acc, f1, loss")
    # t_dev_acc, t_f1, t_dev_loss = evaluate(net, criterion, train_loader, gpu)
    # print(t_dev_acc, t_f1, t_dev_loss)

    