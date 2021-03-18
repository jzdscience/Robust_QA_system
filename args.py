import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)  # have default
    parser.add_argument('--num-epochs', type=int, default=3) # have default
    parser.add_argument('--lr', type=float, default=3e-5) # have default
    parser.add_argument('--num-visuals', type=int, default=10) # have default
    parser.add_argument('--seed', type=int, default=42) # have default
    parser.add_argument('--save-dir', type=str, default='save/') # have default
#     parser.add_argument('--train', action='store_true')
#     parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train-datasets', type=str, default='squad,nat_questions,newsqa') # have default
    parser.add_argument('--run-name', type=str, default='multitask_distilbert') # have default
    parser.add_argument('--recompute-features', action='store_true') 
    parser.add_argument('--train-dir', type=str, default='datasets/indomain_train') # have default
    parser.add_argument('--val-dir', type=str, default='datasets/indomain_val') # have default
    parser.add_argument('--eval-dir', type=str, default='datasets/oodomain_test') # have default
    parser.add_argument('--eval-datasets', type=str, default='race,relation_extraction,duorc') # have default
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--do-finetuning', action='store_true') #new
    parser.add_argument('--sub-file', type=str, default='') # have default
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)  # have default
    parser.add_argument('--adv-training', action='store_true') 
    parser.add_argument('--class-number', type=int, default=3)  # new, have default
    parser.add_argument('--dis-lambda', type=float, default=0.5)  # new, have default
    parser.add_argument('--freeze', action='store_true')   # new
    parser.add_argument('--pretrained-model', type=str, default=None)   # new
    args = parser.parse_args()
    return args
