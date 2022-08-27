import numpy as np
import torch
import json
import sys
import pandas as pd 
import os
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch import cuda
import unidecode
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from numpy.linalg import norm
from numpy import dot
from tqdm import tqdm


BATCH_SIZE = 32
device = "cuda" if cuda.is_available() else "cpu"

languages_map = {
    'en': {"label": "English", 'id': 0},
    'hi': {"label": "Hindi", 'id': 1},
    'te': {"label": "Telugu", 'id': 2},
    'bn': {"label": "Bengali", 'id': 3},
    'pa': {"label": "Punjabi", 'id': 4},
    'or': {"label": "Odia", 'id': 6},
    'as': {"label": "Assamese", 'id': 7},
    'gu': {"label": "Gujarati", 'id': 8},
    'mr': {"label": "Marathi", 'id': 9},
    'kn': {"label": "Kannada", 'id': 10},
    'ta': {"label": "Tamil", 'id': 11},
    'ml': {"label": "Malayalam", 'id': 12}
}

def prepare_tokenizer(tokenizer):
    new_tokens = ['<H>', '<R>', '<T>', '<QR>', '<QT>', '<S>']
    new_tokens_vocab = {}
    new_tokens_vocab['additional_special_tokens'] = []
    for idx, t in enumerate(new_tokens):
        new_tokens_vocab['additional_special_tokens'].append(t)
    num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)

def get_nodes(n):
    n = n.strip()
    n = n.replace('(', '')
    n = n.replace('\"', '')
    n = n.replace(')', '')
    n = n.replace(',', ' ')
    n = n.replace('_', ' ')
    n = unidecode.unidecode(n)
    return n


def get_relation(n):
    n = n.replace('(', '')
    n = n.replace(')', '')
    n = n.strip()
    n = n.split()
    n = "_".join(n)
    return n


def linear_fact_str(fact, enable_qualifiers=True):
    fact_str = ['<R>', get_relation(fact[0]).lower(), '<T>', get_nodes(fact[1]).lower()]
    qualifier_str = [' '.join(['<QR>', get_relation(x[0]).lower(), '<QT>', get_nodes(x[1]).lower()]) for x in fact[2]]
    if enable_qualifiers and len(fact[2])>0:
        fact_str += [' '.join(qualifier_str)]
    return fact_str

def process_facts(facts, entity, section, language):
    """ linearizes the facts on the encoder side """
    linearized_facts = []
    for i in range(len(facts)):
        linearized_facts += linear_fact_str(facts[i], enable_qualifiers=True)
    processed_facts_str = ' '.join(linearized_facts)
    pre_string = "<H> %s %s <S> %s" % (entity.lower(), processed_facts_str, section)
    return "generate %s : %s" % (languages_map[language]['label'].lower(), pre_string)

class JsonL(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, lang, isTarget = True):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targ = isTarget
        self.lang = lang

    def __getitem__(self, index):
        sent  = self.data.sentence[index]
        factlist = self.data.facts[index]
        entity = self.data.entity_name[index]
        section = self.data.native_sentence_section[index]
        fact_str = process_facts(factlist,entity,section,self.lang)    
        
        return {
            "fact_str" : fact_str,
            "sent" : sent
        }
            
    def __len__(self):
        return self.len

def collate(batch):
    fact_str_list = []
    sent_list = []
    ref_list = []
    
    for i in batch :
        fact_str_list.append(i['fact_str'])
        sent_list.append(i['sent'])
        
    return {'fact_str':fact_str_list,'sent' : sent_list}

def load_jsonl(file_path):
    res = []
    with open(file_path, encoding='utf-8') as dfile:
        for line in dfile.readlines():
            res.append(json.loads(line.strip()))
    return res

def load_checkpoint(model, tokenizer):
    PATH = "/scratch/tabhishek/epoch=29.ckpt" 
    model.resize_token_embeddings(len(tokenizer))
    checkpoint = torch.load(PATH)

    new_state_dict = {}
    for ij in checkpoint['state_dict']:
        if ij[:6] == 'model.':
            newkey = ij[6:]
            new_state_dict[newkey] = checkpoint['state_dict'][ij]
        else:
            continue
            print (ij)
        
    del new_state_dict['lm_head.weight']
    model.load_state_dict(new_state_dict)
    print("loaded checkpoint successfully")

def get_order(filename, model, tokenizer):
    reader  = load_jsonl(filename)
    jsonl_data = pd.DataFrame(reader)
    data_set = JsonL(jsonl_data,tokenizer,518, 'hi')
    load_params = {"batch_size": BATCH_SIZE, "shuffle": False, "num_workers": 0}
    data_loader = DataLoader(data_set, sampler=SequentialSampler(data_set),**load_params, collate_fn = collate)
    pred_order = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            sent = data['sent']
            fact_str = data['fact_str']
            
            encoded_sentence = tokenizer.batch_encode_plus(sent, return_tensors="pt",padding=True)
            encoded_facts = tokenizer.batch_encode_plus(fact_str, return_tensors="pt",padding=True)
            output = model(
                input_ids=encoded_facts["input_ids"].to(device,dtype=torch.long),
                attention_mask=encoded_facts["attention_mask"].to(device,dtype=torch.long),
                decoder_input_ids=encoded_sentence["input_ids"].to(device,dtype=torch.long),
                decoder_attention_mask=encoded_sentence["attention_mask"].to(device,dtype=torch.long),
                output_attentions=True  
            )
            
            batch_fin_crossattention = output.cross_attentions[-1].mean(dim=1)
            for itr in range(len(batch_fin_crossattention)):    
                fin_crossattention = batch_fin_crossattention[itr]
                stflag = -1
                currfact = -1
                factpos = []
                factpos2 = []
                
                tokenized_sent = tokenizer.tokenize(sent[itr])
                tokenized_fact = tokenizer.tokenize(fact_str[itr])
                sent_len = len(tokenized_sent)
                for wordval in range(len(tokenized_fact)):
                    tokenized_curr = tokenized_fact[wordval]
                    if (tokenized_curr) == "<R>" and stflag == -1:
                        stflag = 1
                        sumval = 0 
                        fcount = 0
                        maxpos = 0
                        maxscore = -99
                        continue
                    elif (tokenized_curr) == "<R>" and stflag != -1:
                        factpos.append(maxpos)
                        factpos2.append(sumval/fcount)
                        stflag = 1
                        sumval = 0 
                        fcount = 0
                        maxpos = 0
                        maxscore = -99
                        continue 
                    elif (tokenized_curr) == "<QR>":
                        continue
                    elif (tokenized_curr) == "<S>":
                        factpos.append(maxpos)
                        factpos2.append(sumval/fcount)
                        stflag = 0
                        continue

                    max_fact = fin_crossattention[:,wordval][:sent_len].argmax()
                    argval = fin_crossattention[:,wordval][:sent_len].max()
                    intval = np.array(max_fact.to('cpu'))
                    # tempind = encoded_facts.word_ids()[intval]
                    if stflag == 1:
                        sumval += (intval+1)
                        fcount += 1
                        if argval > maxscore:
                            maxpos = (intval+1)
                            maxscore = argval
                final_ordering =  np.argsort(factpos2)
                pred_order.append([int(tv) for tv in list(final_ordering)])
    return pred_order

def store_jsonl(res, file_path):
    with open(file_path, 'w', encoding='utf-8') as dfile:
        for line in res:
            json.dump(line, dfile, ensure_ascii=False)
            dfile.write('\n')

def process(dir_path):
    model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir="/tmp/huggingface")
    model = AutoModel.from_pretrained(model_name,output_hidden_states=True, cache_dir="/tmp/huggingface")
    prepare_tokenizer(tokenizer)
    load_checkpoint(model, tokenizer)
    print("loaded model successfully !!!")
    model = model.to(device)
    # for data_type in ["test", "val", "train"]:
    for data_type in ["test"]:
        count=0
        print("working on %s .. " % data_type)
        for lang_dir_name in os.listdir(dir_path):
            count+=1
            if lang_dir_name not in languages_map:
                continue
            print("[%d / %d] working on %s .. " % (count, len(os.listdir(dir_path)), lang_dir_name))
            file_path = os.path.join(dir_path, lang_dir_name, "%s.jsonl" % data_type)
            pred_order = get_order(file_path, model, tokenizer)
            old_data = load_jsonl(file_path)
            assert len(pred_order)==len(old_data), "length of pred_order is different from old_data"
            for idx, item in enumerate(old_data):
                assert len(item['facts']) == len(pred_order[idx]), "unequal length of pred and actual facts encountered"
                item['order'] = pred_order[idx]
            store_jsonl(old_data, file_path)
            print('--'*30)
        print('=='*30)

if __name__ == "__main__":
    process(sys.argv[1])