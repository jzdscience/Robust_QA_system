import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter


from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args

from tqdm import tqdm

## customized
import advModel

def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples



def prepare_train_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # # 0 -14999 
    ## 242304 
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    # 0 -14999 
    ## 242304 
    offset_mapping = tokenized_examples["offset_mapping"]
    
#     print(len(sample_mapping))
#     print(len(offset_mapping))

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    tokenized_examples['labels'] = []
    
    # add labels
#     tokenized_examples['labels'] = []
    
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
#         print(len(tokenized_examples["input_ids"]))
        
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        
#         labels = dataset_dict['doc'][i]

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        
        ## add labels
        tokenized_examples['labels'].append(dataset_dict['doc'][sample_index])
        
        
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1
    
    # make the labels into integers
    doc_map = {v: k for k,v in enumerate(set(dataset_dict['doc']))}
    tokenized_examples['labels']  = [doc_map[ele] for ele in tokenized_examples['labels']]
    
#     print(len(set(tokenized_examples['labels'])))
#     print(len(tokenized_examples['labels']))
#     print(tokenized_examples['labels'][-10:])
    
    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples



def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    #TODO: cache this if possible
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples



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

    def evaluate_baseline(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train_baseline(self, model, train_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    
                    # tuple of 2
                    outputs = model(input_ids, attention_mask=attention_mask, 
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                    loss = outputs[0]
  
                    loss.backward()  #added sum here
                    
                    optim.step()
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate_baseline(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                    global_idx += 1
        return best_scores
    
    def evaluate_adv(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                
#                 outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
#                 start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # Forward
                start_logits, end_logits = model(input_ids, attention_mask=attention_mask)
                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results
    
    def train_adv(self, model, train_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        qa_optimizer = AdamW(model.parameters(), lr=self.lr)
        dis_optimizer = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)
        
        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            avg_qa_loss = 0
            avg_dis_loss = 0
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
#                     optim.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    
                    labels = batch['labels'].to(device)
                    
                    # remove unnecessary pad token
                    seq_len = torch.sum(torch.sign(input_ids), 1)
                    max_len = torch.max(seq_len)

                    input_ids = input_ids[:, :max_len].clone()
                    attention_mask = attention_mask[:, :max_len].clone()
#                     seg_ids = seg_ids[:, :max_len].clone()
                    start_positions = start_positions.clone()
                    end_positions = end_positions.clone()
                    labels = labels.clone()

#                     if self.args.use_cuda:
#                         input_ids = input_ids.cuda(self.args.gpu, non_blocking=True)
#                         input_mask = input_mask.cuda(self.args.gpu, non_blocking=True)
#                         seg_ids = seg_ids.cuda(self.args.gpu, non_blocking=True)
#                         start_positions = start_positions.cuda(self.args.gpu, non_blocking=True)
#                         end_positions = end_positions.cuda(self.args.gpu, non_blocking=True)
#                     all_labels.append(i * torch.ones_like(start_positions))
    
                    qa_loss = model(input_ids, attention_mask,
                                         start_positions, end_positions, labels = None, 
                                         dtype="qa")
        
#                     print('qa_loss', qa_loss) ## 0-d tensor, seems right
            
                    qa_loss = qa_loss.mean()
                    qa_loss.backward()

                    # update qa model
                    avg_qa_loss = self.cal_running_avg_loss(qa_loss.item(), avg_qa_loss)
                    qa_optimizer.step()
                    qa_optimizer.zero_grad()

                    # update discriminator
                    dis_loss = model(input_ids, attention_mask,
                                          start_positions, end_positions, labels = labels, dtype="dis")
                    dis_loss = dis_loss.mean()
                    dis_loss.backward()
                    avg_dis_loss = self.cal_running_avg_loss(dis_loss.item(), avg_dis_loss)
                    dis_optimizer.step()
                    dis_optimizer.zero_grad()

                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=qa_loss.item())
                    
                    tbx.add_scalar('train/NLL', qa_loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate_adv(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
#                             self.save(model.bert)  ## ?????????????????? should I save this way?
                            ## not sure how to save as BERT pretrained model
                            torch.save(model, self.path + '/pytorch_model.bin')
                    global_idx += 1
        return best_scores                    

    @staticmethod
    def cal_running_avg_loss(loss, running_avg_loss, decay=0.99):
        if running_avg_loss == 0:
            return loss
        else:
            running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
            return running_avg_loss    

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
    return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict

## this is the main function
def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)
    ## it is from hugging face
    if args.adv_training:
        model = advModel.DomainQA(num_classes=3, 
                                   hidden_size=768,
                                   num_layers=3, 
                                   dropout=0.1, 
                                   dis_lambda=0.5,
                                   concat=False, 
                                   anneal=False)
    else:
        model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    
    # tokenizer is distilBertTokenizer instance
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # if training 
    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
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
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        
        # create data iterable 
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        #call trainer to train
        if args.adv_training: 
            best_scores = trainer.train_adv(model, train_loader, val_loader, val_dict)
        else:
            best_scores = trainer.train_baseline(model, train_loader, val_loader, val_dict)
        
    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # if path contain test then split is test
        
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        
        trainer = Trainer(args, log)
        
        # load model  from checkpoint
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        
        ### should I replace the model here????
        #########
        if args.adv_training:
            if torch.cuda.is_available():
                model = torch.load(checkpoint_path + '/pytorch_model.bin')
            else:
                model = torch.load(checkpoint_path + '/pytorch_model.bin', map_location=torch.device('cpu'))
                
        else:    
            model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        #########
        model.to(args.device)
        
        # load the test dataset
        eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        
        if args.adv_training: 
            eval_preds, eval_scores = trainer.evaluate_adv(model, eval_loader,
                                                       eval_dict, return_preds=True,
                                                       split=split_name)
        else:
            eval_preds, eval_scores = trainer.evaluate_baseline(model, eval_loader,
                                                       eval_dict, return_preds=True,
                                                       split=split_name)
        # Write log file
        ## print the EM; F1 in the log_test
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        
        # write result to test_
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()
