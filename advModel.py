import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_pretrained_bert import BertModel, BertConfig
# from utils import kl_coef
from util import kl_coef

from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertConfig

class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=6, input_size=768 * 2,
                 hidden_size=768, num_layers=3, dropout=0.1):
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
#             hidden_layers.append(nn.Sequential(   
#                 nn.Linear(input_dim, hidden_size),
# #                 nn.BatchNorm1d(num_features=hidden_size),
#                 nn.ReLU(), 
#                 nn.Dropout(dropout)
#             ))
#         hidden_layers.append(nn.Linear(hidden_size, num_classes))
#         self.hidden_layers = nn.ModuleList(hidden_layers)

            hidden_layers.append(nn.Sequential(OrderedDict([   
                ('linear', nn.Linear(input_dim, hidden_size)),
                ('relu', nn.ReLU()), 
                ('drop', nn.Dropout(dropout))
            ])))
        hidden_layers.append(nn.Sequential(OrderedDict([  ('output', nn.Linear(hidden_size, num_classes))])))
        self.hidden_layers = nn.ModuleList(hidden_layers)
            


    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)   # x -- torch.Size([16, 768])
            
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob


class DomainQA(nn.Module):
    def __init__(self, args, bert_name_or_config = "distilbert-base-uncased", num_classes=6, hidden_size=768,
                 num_layers=3, dropout=0.1, dis_lambda=0.5, concat=False, anneal=False, pre_trained = None):
        super(DomainQA, self).__init__()
        
        if pre_trained is None:
            self.config = DistilBertConfig.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
            self.bert = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased', config=self.config )
        
        else:
            self.bert = DistilBertForQuestionAnswering.from_pretrained(args.pretrained_model)
            
        self.qa_outputs = nn.Linear(hidden_size, 2)   # 768 *2
        # init weight
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.qa_outputs.bias.data.zero_()
        if concat:
            input_size = 2 * hidden_size
        else:
            input_size = hidden_size   # 768
        self.discriminator = DomainDiscriminator(num_classes, input_size, hidden_size, num_layers, dropout)

        self.num_classes = num_classes
        self.dis_lambda = dis_lambda
        self.anneal = anneal
        self.concat = concat
        self.sep_id = 102

    # only for prediction
    def forward(self, input_ids,  attention_mask,
                start_positions=None, end_positions=None, labels=None,
                dtype=None, global_step=22000):
        if dtype == "qa":
            #      
            qa_loss = self.forward_qa(input_ids,  attention_mask,
                                      start_positions, end_positions, global_step)
            return qa_loss

        elif dtype == "dis":
            assert labels is not None
            dis_loss = self.forward_discriminator(input_ids,  attention_mask, labels)
            return dis_loss

        else:
            # use for evaluate
            
#             sequence_output, _ = self.bert(input_ids,  attention_mask)
            _, _, hidden_states = self.bert(input_ids,  attention_mask, return_dict = False)
            # try: take the last hidden states
            sequence_output = hidden_states[-1]
            #              768*2         16*384*768     
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
#             print('interesting else is called here')
            return start_logits, end_logits
    
    # ###       torch.Size([16, 384]) ([16, 384])   ([16, 384])     ([16])             ([16])
    def forward_qa(self, input_ids,  attention_mask, start_positions, end_positions, global_step):
        
        ### initialize bert  
        #### original bert
        #### sequence_output = torch.Size([16, 384, 768]) last_hidden_state   _ = torch.Size([16, 768]) pooler_output
        
        #### distilBert output is (start_scores torch.Size([16, 384]) , end_scores torch.Size([16, 384])) tuple,
        ####                      tuple of 7 tensors
        _, _, hidden_states = self.bert(input_ids,  attention_mask, return_dict = False)
        # try: take the last hidden states
        sequence_output = hidden_states[-1]
        
        cls_embedding = sequence_output[:, 0]   # torch.Size([16, 768])
        
        if self.concat:
            sep_embedding = self.get_sep_embedding(input_ids, sequence_output)
            hidden = torch.cat([cls_embedding, sep_embedding], dim=1)
        else:
            #   take the first token of the 384 (CLS?) token
            hidden = sequence_output[:, 0]  # [b, d] : [CLS] representation    # torch.Size([16, 768])

        
        ######################## here we called  discriminator !! 
        log_prob = self.discriminator(hidden)     ##  log_prob torch.Size([16, 6])
        ############################################################

        
        targets = torch.ones_like(log_prob) * (1 / self.num_classes)
        # As with NLLLoss, the input given is expected to contain log-probabilities
        # and is not restricted to a 2D Tensor. The targets are given as probabilities
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        if self.anneal:
            self.dis_lambda = self.dis_lambda * kl_coef(global_step)
        kld = self.dis_lambda * kl_criterion(log_prob, targets)
        
        #              768*2      [16, 384, 768]   
        logits = self.qa_outputs(sequence_output)   # logits: torch.Size([16, 384, 2])
        
        start_logits, end_logits = logits.split(1, dim=-1)  ###  split to two equal portion by last dimension  [16, 384,1]
        start_logits = start_logits.squeeze(-1)  # 16*384
        end_logits = end_logits.squeeze(-1)  # 16*384
        

#         print(start_positions.size())  # tensor 16,
#         print(end_positions.size())  # tensor 16,

        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
            
        # sometimes the start/end positions are outside our model inputs, we ignore these terms ???????
        ignored_index = start_logits.size(1)  # 384
        
        # clamping is just make sure the number is between 0 and 384
        start_positions.clamp_(0, ignored_index)  # torch.Size([16])
        end_positions.clamp_(0, ignored_index)  # torch.Size([16])
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)   # CrossEntropyLoss()
        
        #                     16*384          torch.Size([16]) (int for position)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        qa_loss = (start_loss + end_loss) / 2
        total_loss = qa_loss + kld
        return total_loss

    def forward_discriminator(self, input_ids,  attention_mask, labels):
        with torch.no_grad():
            
#             sequence_output, _ = self.bert(input_ids,  attention_mask)
            
            _, _, hidden_states = self.bert(input_ids,  attention_mask, return_dict = False)
            # try: take the last hidden states
            sequence_output = hidden_states[-1]
            
            #  # torch.Size([16, 768])
            cls_embedding = sequence_output[:, 0]  # [b, d] : [CLS] representation
            if self.concat:
                sep_embedding = self.get_sep_embedding(input_ids, sequence_output)
                hidden = torch.cat([cls_embedding, sep_embedding], dim=-1)  # [b, 2*d]
            else:
                hidden = cls_embedding
        #                                detach: remove from grad computation
        log_prob = self.discriminator(hidden.detach())
        criterion = nn.NLLLoss()
        loss = criterion(log_prob, labels)

        return loss

    def get_sep_embedding(self, input_ids, sequence_output):
        batch_size = input_ids.size(0)
        sep_idx = (input_ids == self.sep_id).sum(1)
        sep_embedding = sequence_output[torch.arange(batch_size), sep_idx]
        return sep_embedding
    
