import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForMaskedLM
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter


from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args

from tqdm import tqdm

# ## customized
# import advModel
def mask_tokens(inputs, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone().unsqueeze(0)
    inputs = inputs.unsqueeze(0)
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
#     import pdb; pdb.set_trace()
    inputs = inputs.squeeze()
    labels = labels.squeeze()
    return inputs, labels


def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=False,
                                   return_offsets_mapping=False,
                                   padding='max_length', return_tensors="pt")
    
    input_ids = tokenized_examples["input_ids"]
    
    tokenized_examples["input_ids"] = []
    tokenized_examples['labels'] = []
    for i, input_id in enumerate(tqdm(input_ids)):

        inputs, labels =  mask_tokens(input_id, tokenizer)

        tokenized_examples['input_ids'].append(inputs)
        
        tokenized_examples['labels'].append(labels)
    
    print('tokenization complete')
    return tokenized_examples


def prepare_train_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=False,
                                   return_offsets_mapping=False,
                                   padding='max_length', return_tensors="pt")

    input_ids = tokenized_examples["input_ids"]
    
    tokenized_examples["input_ids"] = []
    tokenized_examples['labels'] = []
    
    for i, input_id in enumerate(tqdm(input_ids)):

        inputs, labels =  mask_tokens(input_id, tokenizer)

        tokenized_examples['input_ids'].append(inputs)
        
        tokenized_examples['labels'].append(labels)
    
    print('tokenization complete')
    return tokenized_examples



def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    #TODO: cache this if possible
    cache_path = f'{dir_name}/{dataset_name}_encodings_mlm.pt'
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
# not saving
#         util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples


def get_dataset(args, datasets, data_dir, tokenizer, split_name):
    # ['NewsQA', 'Squad', ...]
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    # iterate strings of datasets (merged together)
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')

        ## adding encoding from dataset to dictionary
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
        
    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
#     import pdb; pdb.set_trace()
    return util.MLMDataset(data_encodings, train=True), dataset_dict

#TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate_baseline(self, model, data_loader):
        
        device = self.device
        
        val_loss = 0
        global_idx = 0
        model.eval()
        
        with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(inputs, attention_mask= attention_mask, labels=labels)
                loss = outputs[0]
                val_loss += loss.item()
                global_idx += 1
                
        final_loss = val_loss/global_idx
        print('current validation loss: ', final_loss)
        
        return final_loss
        

    def train_baseline(self, model, train_dataloader, eval_dataloader):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        
        tbx = SummaryWriter(self.save_dir)
        
        tr_loss = 0
        
        best_val_loss = 999
        
        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
#                 import pdb; pdb.set_trace()
                for batch in train_dataloader:
#                     import pdb; pdb.set_trace()
                    optim.zero_grad()
                    model.train()
                    
                    ## MLM
             
                    
                    inputs = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                                                
#                     import pdb; pdb.set_trace() 
                                                
                    outputs = model(inputs, attention_mask= attention_mask, labels=labels)
 
                    loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                    loss.backward()  #added sum here
#                     import pdb; pdb.set_trace()
                    tr_loss += loss.item()
                    
                    optim.step()
                    
                    progress_bar.update(len(inputs))
                    
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('training loss', loss.item(), global_idx)
                    
                    ## calling for evaluation
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        
                        current_val_loss= self.evaluate_baseline(model, eval_dataloader)
                        
                        results_str = f'the current validation loss is {current_val_loss}'
                        
                        self.log.info('Visualizing in TensorBoard...')
                        
#                         tbx.add_scalar(f'val', current_val_loss.item(), global_idx)
                            
                        self.log.info(f'Eval {results_str}')
#                         if self.visualize_predictions:
#                             util.visualize(tbx,
#                                            pred_dict=preds,
#                                            gold_dict=val_dict,
#                                            step=global_idx,
#                                            split='val',
#                                            num_visuals=self.num_visuals)
                        if current_val_loss < best_val_loss:
                            best_val_loss = current_val_loss
                            self.save(model)
                            
                    global_idx += 1
        print('average training loss: ', tr_loss/global_idx)
        return best_val_loss




## this is the main function
def main():
    # define parser and arguments
    args = get_train_test_args()
    print(args)
    util.set_seed(args.seed)
    
    # tokenizer is distilBertTokenizer instance
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # if training, then start a new Domain QA model or a distrilBert Model
    if args.do_train:

        model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased') 
#         import pdb; pdb.set_trace()
        # make save/ folder
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        # get save/run_name_xxx folder
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        
        # save log for tensorboard
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # initialize Trainer instance
        trainer = Trainer(args, log)
        
        # get_dataset --> read_and_process --> prepare_train_data (provide encoding/tokenization): return a QAdataset instance
        train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
        
#         log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        
        # create data iterable 
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))

        #call trainer to train
        print('beging training')
        best_scores = trainer.train_baseline(model, train_loader, val_loader)
            

if __name__ == '__main__':
    main()
