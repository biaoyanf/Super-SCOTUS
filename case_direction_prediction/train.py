import time 

from torch.utils.data import DataLoader
from CaseDataset import CaseDataset
from CaseClassifier import CaseClassifier
from sklearn.metrics import f1_score,accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import pyhocon
import os
import errno

import json  

import random
import numpy as np 

import matplotlib.pyplot as plt

def plot_learning_curve(train_loss = None, dev_loss = None, save_path= "./"): 
    if train_loss: 
      
        epoch_index = [i+1 for i in range(len(train_loss))] 
        
        plt.plot(epoch_index, train_loss, label = "Train") 
        
    if dev_loss:  
        epoch_index = [i+1 for i in range(len(dev_loss))] 
        plt.plot(epoch_index, dev_loss, label = "Dev") 
    
    plt.title("Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
#     plt.show()
    plt.xticks(epoch_index, epoch_index)
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss.png"))
    plt.close()

def plot_evaluations(train_eval = None, dev_eval = None, save_path= "./"): 
    if not train_eval: 
        
        epoch_index = [i+1 for i in range(len(train_eval))] 
    
        plt.plot(epoch_index, train_eval, label = "Train") 
    
    if dev_eval:  
        epoch_index = [i+1 for i in range(len(dev_eval))] 

        plt.plot(epoch_index, dev_eval, label = "Dev") 
    
    plt.title("Evaluation")
    plt.xlabel("Step")
    plt.ylabel("Macro F1")
    plt.tight_layout()
#     plt.show()
    plt.xticks(epoch_index, epoch_index)
    plt.legend()
    plt.savefig(os.path.join(save_path, "f1.png"))
    plt.close()


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


def train(net, config, criterion, opti, train_loader, dev_loader, max_eps, gpu):
    best_acc = 0 
    best_f1 = 0
    best_macrof1 = 0
    st = time.time()

    train_loss_list = []
    dev_loss_list = []

    train_eval_list = []
    dev_eval_list = [] 

    temp_train_loss = [] 
    global_steps = 0 
    print("start training...")
    for ep in range(max_eps): 
        net.train() # need to define here again as it will be set to eval() in evaluate function 
        # temp_train_eval = []
        print("training ep: ", ep)
        for it, (seq, attn_masks, labels, ids) in enumerate(train_loader):
            #Clear gradients
            # print(it)
            # print("type(seq):", type(seq))
            # print("type(attn_masks):", type(attn_masks)) 
            # print("type(labels):" , type(labels))
            # print("seq.size(): ", seq.size() )
            # print('\n')

            opti.zero_grad()  
            #Converting these to cuda tensors
            if gpu >= 0:
                seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)
            else:
                seq, attn_masks, labels = seq.cpu(), attn_masks.cpu(), labels.cpu()
            # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

            #Obtaining the logits from the model
            logits = net(seq, attn_masks)
            # print("logits ", logits)
            # print("lables: ", labels)
            #Computing loss 

            # print(logits.squeeze(-1))
            loss = criterion(logits.squeeze(-1), labels.float())

            temp_train_loss.append(loss.item()) 

            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            opti.step()
              
            global_steps +=1  
            
            if global_steps % config["eval_steps"] == 0:  # go into evaluation 
                print("Step {} evaluating...".format(global_steps))
        
                # dev_macrof1, dev_acc, dev_f1, dev_loss = evaluate_random_sample(net, criterion, dev_loader, gpu, global_steps)
                dev_macrof1, dev_acc, dev_f1, dev_loss = evaluate_file(net, criterion, dev_loader, "dev", gpu)
                # train_macrof1, train_acc, train_f1, train_loss = evaluate_random_sample(net, criterion, train_loader, gpu, global_steps)
                # train_macrof1, train_acc, train_f1, train_loss = evaluate(net, criterion, train_loader, gpu)

                #  the ploting part  
                print("the ave and eval train loss: ", sum(temp_train_loss)/len(temp_train_loss))  


                train_loss_list.append(sum(temp_train_loss)/len(temp_train_loss))

                temp_train_loss = []

                dev_loss_list.append(dev_loss) 

                plot_learning_curve(train_loss_list, dev_loss_list, config["log_dir"])
                

                # train_eval_list.append(sum(temp_train_eval)/len(temp_train_eval)) 
                # train_eval_list.append(train_f1)
                dev_eval_list.append(dev_macrof1)

                plot_evaluations([], dev_eval_list, config["log_dir"])
                #  done with ploting part 


                print("Step {} complete! Development Accuracy: {}; f1_score: {}; Macro f1_score: {}; Development Loss: {}".format(global_steps, dev_acc, dev_f1, dev_macrof1, dev_loss))
                
                if dev_macrof1 >= best_macrof1:
                    print("Best development Macro F1 improved from {} to {}, saving model...".format(best_macrof1, dev_macrof1))
                    best_macrof1 = dev_macrof1 

                    torch.save(net.state_dict(), './{}/sstcls_best.dat'.format(config["log_dir"]))
                    # torch.save(net.state_dict(), './{}/saved_weights_LSTM.pt'.format(config["log_dir"]))

                net.train() # need to define here again as it will be set to eval() in evaluate function 

            # if global_steps > config["max_traning_steps"]: break 
      
            # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

def get_accuracy_from_logits(logits, labels):
    # probs = torch.sigmoid(logits.unsqueeze(-1)) 

    soft_probs = (logits > 0.5).float()
    # print("logits ", logits)
    # print("lables: ", labels)
    # print("soft_probs: ", soft_probs)
    # pred_arg_max = torch.argmax(probs, dim=1) # now gfet a list [1, 2, 0, ...]  -> to get the max index
    # pred_arg_max = torch.squeeze(pred_arg_max)
    # ground_arg_max = torch.argmax(labels, dim=-1)
    # print(pred_arg_max)
    # print(pred_arg_max.tolist(), labels.tolist())
    # print((pred_arg_max == labels).float().tolist())
    # acc = (pred_arg_max == labels).float().mean()
    
    flat_soft_probs = torch.reshape(soft_probs, (-1,))
    # flat_pred_arg_max = torch.reshape(pred_arg_max, (-1,))
    flat_labels = torch.reshape(labels, (-1,))
    f1 = f1_score(flat_labels.cpu(), flat_soft_probs.cpu(), average = "micro") 
    macrof1 = classification_report(flat_labels.cpu(), flat_soft_probs.cpu(),output_dict=True)["macro avg"]["f1-score"]


    acc = accuracy_score(flat_labels.cpu(), flat_soft_probs.cpu()) 
    
    # if f1 == 1.0:
    # print("prediction: ", flat_pred_arg_max)
    # print("gold_label: ", flat_labels)
    # print("overlap: ", set(flat_labels)-set(flat_pred_arg_max), set(flat_pred_arg_max)-set(flat_labels))
    # print("\n")

    return macrof1, acc, f1



def evaluate(net, criterion, dataloader, gpu):
    net.eval()

    mean_acc, mean_loss = 0, 0
    mean_f1 = 0
    mean_macrof1 = 0 
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
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item() 
            # tmp_macrof1, tmp_acc, tmp_f1 = get_accuracy_from_logits(logits, labels)
            # mean_acc += tmp_acc
            # mean_f1 += tmp_f1
            # mean_macrof1 += tmp_macrof1
            count += 1

            golds = torch.cat([golds, labels], 0) 
            preds = torch.cat([preds, logits], 0) 
            
            # torch.cuda.empty_cache() 
            
            # store_prediciton(pred_results, ids, logits) 
    # print(pred_results)
    # with open("./{}/pred_{}.txt".format(config["log_dir"], file_type), "w") as fw: 
    #     fw.write(json.dumps(pred_results))
    # print("preds: ", preds) 
    # print("golds: ", golds)
    macrof1, acc, f1 = get_accuracy_from_logits(preds, golds)  

    return macrof1, acc, f1,  mean_loss / count

    # return mean_macrof1 / count, mean_acc / count, mean_f1 / count,  mean_loss / count

    

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
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item() 
            # tmp_macrof1, tmp_acc, tmp_f1 = get_accuracy_from_logits(logits, labels)
            # mean_acc += tmp_acc
            # mean_f1 += tmp_f1
            # mean_macrof1 += tmp_macrof1
            count += 1

            golds = torch.cat([golds, labels], 0) 
            preds = torch.cat([preds, logits], 0) 
            
            store_prediction(pred_results, ids, logits, labels) 
    # print(pred_results)
    with open("./{}/pred_{}.txt".format(config["log_dir"], file_type), "w") as fw: 
        fw.write(json.dumps(pred_results))
    # print("preds: ", preds) 
    # print("golds: ", golds)
    macrof1, acc, f1 = get_accuracy_from_logits(preds, golds)  

    return macrof1, acc, f1,  mean_loss / count

    # return mean_macrof1 / count, mean_acc / count, mean_f1 / count,  mean_loss / count 


def store_prediction(pred_results, ids, logits, labels):
    # print("logits", logits)

    # probs = torch.sigmoid(logits.unsqueeze(-1))
    # pred_arg_max = torch.argmax(probs, dim=1) # now gfet a list [1, 2, 0, ...]  -> to 
    # pred_arg_max = torch.squeeze(pred_arg_max)
    # flat_pred_arg_max = torch.reshape(pred_arg_max, (-1,))
    
    # soft_probs = (logits> 0.5).float()
    soft_probs = logits.float()
    flat_soft_probs = torch.reshape(soft_probs,  (-1,))

    if flat_soft_probs.size() == torch.Size([]):  # after the squeeze pred_arg_max become an int as the batch size become 1. so we need to make it as a list again 
        # print("pred_arg_max: ", pred_arg_max)
        # print("pred_arg_max.size(): ", pred_arg_max.size())
        flat_soft_probs = [flat_soft_probs.tolist()]  
    else:  
        flat_soft_probs = flat_soft_probs.tolist() 

    labels = labels.tolist() 
    for i, p, l in zip(ids, flat_soft_probs, labels):
        # print(i, pred_results)
        assert i not in pred_results
        pred_results[i] = [p, l] 

def set_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.n_gpu > 0:
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":


    seed = 123
    set_seed(seed) 
    
    config = initialize_from_env()
    name = sys.argv[1] 
    
    
    train_set = CaseDataset(config = config, data_path = config["train_label_path"])
    dev_set = CaseDataset(config = config, data_path = config["dev_label_path"]) 
    test_set = CaseDataset(config = config, data_path = config["test_label_path"]) 
    

    #Creating intsances of training and development dataloaders
    train_loader = DataLoader(train_set, batch_size = config["batch_size"], shuffle = True)
    dev_loader = DataLoader(dev_set, batch_size = config["batch_size"], shuffle = True)
    test_loader = DataLoader(test_set, batch_size = config["batch_size"], shuffle = True) 
    
    print("Done preprocessing training, development and test data.")
    

    # gpu = 5  #gpu ID
    gpu = config["gpu"]

    
    print("Creating the CaseClassifier, initialised with pretrained {}".format(config["model_type"]))



    net = CaseClassifier(config = config) 

    print("Done with creating the CaseClassifier.")
   

    if gpu > -1:
        net.cuda(gpu) #Enable gpu support for the model
    else:
        net.cpu()
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss() 


    opti = optim.Adam(net.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])



    num_epoch = config["num_epoch"]

    #fine-tune the model
    train(net, config, criterion, opti, train_loader, dev_loader, num_epoch, gpu)


    net.load_state_dict(torch.load('./{}/sstcls_best.dat'.format(config["log_dir"])))

    print("Best performance in Dev: acc, f1, macrof1, loss")
    # dev_acc, dev_f1, dev_loss = evaluate(net, criterion, dev_loader, gpu)
    dev_macrof1, dev_acc, dev_f1, dev_loss = evaluate_file(net, criterion, dev_loader, "dev", gpu)
    print(dev_acc, dev_f1, dev_macrof1, dev_loss)

    print("Corresponding result in Train: acc, f1, macrof1, loss")
    t_macrof1, t_dev_acc, t_f1, t_dev_loss = evaluate(net, criterion, train_loader, gpu)
    print(t_dev_acc, t_f1, t_macrof1, t_dev_loss)
    

    print("Corresponding result in Test: acc, f1, macrof1, loss")
    # test_acc, test_f1, test_loss = evaluate(net, criterion, test_loader, gpu)
    test_macrof1, test_acc, test_f1, test_loss = evaluate_file(net, criterion, test_loader, "test", gpu)
    print(test_acc, test_f1, test_macrof1, test_loss)

    