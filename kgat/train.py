import random, os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from models import inference_model
from data_loader import DataLoader
from bert_model import BertForSequenceEncoder
from torch.nn import NLLLoss
import logging

logger = logging.getLogger(__name__)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def correct_prediction(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct


def eval_model(model, validset_reader):
    model.eval()
    correct_pred = 0.0
    for index, data in enumerate(validset_reader):
        inputs, lab_tensor = data
        prob = model(inputs)
        correct_pred += correct_prediction(prob, lab_tensor)
    dev_accuracy = correct_pred / validset_reader.total_num
    return dev_accuracy



def train_model(model, ori_model, args, trainset_reader, validset_reader):
    save_path = args.outdir + '/model'
    best_accuracy = 0.0
    running_loss = 0.0
    t_total = int(
        trainset_reader.total_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    extra_decay = ['weights','gammas','ds']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n in extra_decay], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and n not in extra_decay], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    print([n for n, p in param_optimizer if not any(nd in n for nd in no_decay)])
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)
    #optimizer = optim.Adam(model.parameters(),
    #                       lr=args.learning_rate)
    global_step = 0

    with torch.no_grad():
        print("Original baseline accuracy: ",eval_model(model, validset_reader))
    for epoch in range(int(args.num_train_epochs)):
        global_step = 0
        running_loss = 0
        
        model.train()
        optimizer.zero_grad()
        for index, data in enumerate(trainset_reader):
            # print(model.gammas)
            inputs, lab_tensor = data
            prob = model(inputs)

            loss = F.nll_loss(prob, lab_tensor)
 
            running_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                # ------------- Start New Code ---------------
                # # Clip the gradient norm using nn.utils.clip_grad_norm_()
                # nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                # ------------- End New Code ---------------
                optimizer.step()
                optimizer.zero_grad()
                logger.info('Epoch: {0}, Step: {1}, Loss: {2}'.format(epoch, global_step, (running_loss / global_step)))
                # print(model.weights,model.gammas,model.ds)
            # if global_step % (args.eval_step * args.gradient_accumulation_steps) == 0:
        logger.info('Start eval!')
        with torch.no_grad():
            dev_accuracy = eval_model(model, validset_reader)
            logger.info('Dev total acc: {0}'.format(dev_accuracy))
            if dev_accuracy > best_accuracy:
                best_accuracy = dev_accuracy

                torch.save({'epoch': epoch,
                            'model': ori_model.state_dict(),
                            'best_accuracy': best_accuracy}, save_path + f"model_approach2_epoch{epoch+3}.best.pt")
                logger.info("Saved best epoch {0}, best accuracy {1}".format(epoch, best_accuracy))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patience', type=int, default=20, help='Patience')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--kernel", type=int, default=21, help='Evidence num.')
    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
    parser.add_argument("--max_len", default=130, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--eval_step", default=2000, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--bert_pretrain', required=True)
    parser.add_argument('--postpretrain')
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logger.info('Start training!')

    label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
    logger.info("loading training set")
    trainset_reader = DataLoader(args.train_path, label_map, tokenizer, args,
                                 batch_size=args.train_batch_size)
    logger.info("loading validation set")
    validset_reader = DataLoader(args.valid_path, label_map, tokenizer, args,
                                 batch_size=args.valid_batch_size, test=True)

    logger.info('initializing estimator model')
    bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)
    if args.postpretrain:
        model_dict = bert_model.state_dict()
        pretrained_dict = torch.load(args.postpretrain)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)

    # --------------- START NEW CODE --------------------
    
    # We need to use a dictionary to transfer learning from the original model to our model
    new_dict = {"proj_gat.0.weight":"proj_gat[0].weight",
              "proj_gat.0.bias":"proj_gat[0].bias",
              "proj_gat.2.weight":"proj_gat[2].weight",
              "proj_gat.2.bias":"proj_gat[2].bias"}
              
    ori_model = inference_model(bert_model, args)
    flag = False
    for layer_name, params in ori_model.named_parameters():
        if "pred_model.bert.encoder.layer" in layer_name:
            if layer_name[30:32]=="10" or flag:
                new_name = layer_name[:29]+f"[{layer_name[30:32]}]"+layer_name[32:]
                new_dict[layer_name] = new_name
                flag = True
            else:
                new_name = layer_name[:29]+f"[{layer_name[30]}]"+layer_name[31:]
                new_dict[layer_name] = new_name
                

    # We set in which device we will put the matrices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = torch.load("../checkpoint/kgat/model.best.pt", map_location=device)
    
    for name,params in ori_model.named_parameters():
        print(name)
        if "pred_model.bert.encoder.layer" in name:
            params.requires_grad = False
        if name in trained_model["model"].keys():
            layer = trained_model["model"][name]
            layer = torch.nn.Parameter(layer)

            if name in new_dict.keys():

                name = new_dict[name]

            exec("ori_model." + name + " = layer")
        
    # --------------- END NEW CODE --------------------
    
    print("Success")
    ori_model = ori_model.to(torch.device("cuda"))
    
    # Thankfully, we don't need to parallelise
    # model = nn.DataParallel(ori_model)
    # model = model.to(torch.device("cuda"))

    train_model(ori_model, ori_model, args, trainset_reader, validset_reader)
