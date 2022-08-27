import numpy as np
import torch
import json
import sys
import pandas as pd 
import os 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from torch import cuda
from model.model import FineTuner
import random

lang='kn'
def load_jsonl(file_path):
    res = []
    with open(file_path, encoding='utf-8') as dfile:
        for line in dfile.readlines():
            res.append(json.loads(line.strip()))
    return res
filename = "{lang}_order.jsonl".format(lang=lang)
reader  = load_jsonl(filename)
jsonl_data = pd.DataFrame(reader)

# following code to transliterate the input script to unified devanagari
from indicnlp.transliterate import unicode_transliterate
def get_native_text_from_unified_script(unified_text, lang):
    return unicode_transliterate.UnicodeIndicTransliterator.transliterate(unified_text, lang, "hi")

def process_for_bleu(text, lang):
    native_text = text
    if lang!='en':
        # convert unified script to native langauge text
        native_text = get_native_text_from_unified_script(text, lang)

    native_text = native_text.strip()
    # as input and predicted text are already space tokenized
    native_text = ' '.join([x for x in native_text.split()])
    return native_text

jsonl_data['sentence'] = jsonl_data['sentence'].apply(lambda x: process_for_bleu(x, lang))

# print(jsonl_data.head())
# exit()

device = "cuda" if cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
tokenizer.add_tokens(['S|', 'P|', 'O|'])

model_name = "google/mt5-small" #google/muril-base-cased, bert-base-multilingual-cased, xlm-roberta-base, sentence-transformers/LaBSE
PATH = "model.ckpt" 

# model = AutoModel.from_pretrained(model_name,output_hidden_states=True)

# model.resize_token_embeddings(len(tokenizer))
model = FineTuner.load_from_checkpoint('model.ckpt')


# checkpoint = torch.load(PATH)
# print (checkpoint.keys())

# new_state_dict = {}
# for ij in checkpoint['state_dict']:
#     #     print (ij[:6])
#     if ij[:6] == 'model.':
#         newkey = ij[6:]
#         new_state_dict[newkey] = checkpoint['state_dict'][ij]
#     else :
#         print (ij)
    
# del new_state_dict['lm_head.weight']
# model.load_state_dict(new_state_dict)

def collate(batch):
    
    fact_str_list = []
    sent_list = []
    ref_list = []
    lang_list = []
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    role_ids_list = []
    #'role_ids': role_ids,'labels': labels.squeeze(), 'lang': torch.tensor(lang_id), 'fact_str': fact_str, 'sent': target_text, 'ref_order': ref_order
    ref_flag = True 
    
    try :
        k = batch[0]['ref_order']
    except:
        ref_flag = False
    
    for i in batch :
        fact_str_list.append(i['fact_str'])
        sent_list.append(i['sent'])
        lang_list.append(i['lang'])
        input_ids_list.append(i['input_ids'])
        attention_mask_list.append(i['attention_mask'])
        labels_list.append(i['labels'])
        role_ids_list.append(i['role_ids'])
        
        if ref_flag :
            ref_list.append(i['ref_order'])
        
    
    if ref_flag :
        return {'fact_str':fact_str_list,'sent' : sent_list,'ref_order':ref_list, 'lang': lang_list, 'input_ids': input_ids_list, 'attention_mask': attention_mask_list, 'role_ids': role_ids_list, 'labels': labels_list}
    else:
        return {'fact_str':fact_str_list,'sent' : sent_list, 'lang': lang_list}

class DS(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length):
        # self.df = pd.read_csv(data_path, sep='\t', )
        self.df = jsonl_data
        # self.df = self.df[:len(self.df)]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.languages_map = {
            'en': {"label": "English", 'id': 0},
            'hi': {"label": "Hindi", 'id': 1},
            'te': {"label": "Telugu", 'id': 2}, 
            'bn': {"label": "Bengali", 'id': 3},
            'pa': {"label": "Punjabi", 'id': 4},
            'ur': {"label": "Urdu", 'id': 5}, 
            'or': {"label": "Odia", 'id': 6}, 
            'as': {"label": "Assamese", 'id': 7},
            'gu': {"label": "Gujarati", 'id': 8},
            'mr': {"label": "Marathi", 'id': 9},
            'kn': {"label": "Kannada", 'id': 10},
            'ta': {"label": "Tamil", 'id': 11},
            'ml': {"label": "Malayalam", 'id': 12} 
        }

    def __len__(self):
        return len(self.df)

    def process_facts_shiva(self, facts, entity):
        """ linearizes the facts on the encoder side """
        triples = []
        subject = entity
        # print(facts)

        # triples.append([' <S> '+subject])

        for triple in facts:
            predicate = triple[0]
            object = triple[1]
            triples.append([' S| '+subject, ' P| '+predicate, ' O| '+object])

            for quals in triple[2]:
                qual_subject = object
                qual_predicate = quals[0]
                qual_object = quals[1]
                triples.append([' S| '+qual_subject, ' P| '+qual_predicate, ' O| '+qual_object])

        return triples

    def __getitem__(self, idx):
        # print(type(self.df.iloc[idx]['facts']))
        # input_text = self.process_facts_shiva(eval(self.df.iloc[idx]['facts']))
        entity = self.df.iloc[idx]['entity_name']
        input_text = self.process_facts_shiva(self.df.iloc[idx]['facts'], entity)
        # print(input_text, type(input_text))
        target_text = self.df.iloc[idx]['sentence']
        lang_code = self.df.iloc[idx]['lang']
        lang = self.languages_map[lang_code]['label']
        lang_id = self.languages_map[lang_code]['id']
        prefix = f'rdf to {lang} text: '
        # prefix = f'rdf to en text: '
        try:
            ref_order = self.df.iloc[idx]['order']
        except:
            print("ref_order_not _found")

        input_encoding = self.role_specific_encoding(prefix, input_text)
        # input_encoding = self.tokenizer(input_text, return_tensors='pt', max_length=self.max_source_length ,padding='max_length', truncation=True)
        target_encoding = self.tokenizer(target_text, return_tensors='pt', max_length=self.max_target_length ,padding='max_length', truncation=True)

        input_ids, attention_mask, role_ids, fact_str = input_encoding['input_ids'], input_encoding['attention_mask'], input_encoding['role_ids'], input_encoding['fact_str']
        labels = target_encoding['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100    # for ignoring the cross-entropy loss at padding locations

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'role_ids': role_ids,'labels': labels.squeeze(), 'lang': torch.tensor(lang_id), 'fact_str': fact_str, 'sent': target_text, 'ref_order': ref_order}   
        # squeeze() is needed to remove the batch dimension

    def role_specific_encoding(self, prefix, input_text):
        input_ids = []
        attention_mask = []
        role_ids = []
        fact_str = ''

        prefix_tokenized = self.tokenizer.encode(prefix)[:-1]   # ignoring the eos token at the end
        input_ids.extend(prefix_tokenized)
        attention_mask.extend([1] * len(prefix_tokenized))
        role_ids.extend([0] * len(prefix_tokenized))
        fact_str += prefix
        try:
            # data = json.loads(input_text)
            # data = eval(input_text)
            data = input_text
        except json.decoder.JSONDecodeError:
            print(input_text)
            raise

        for triple in data:
            subject = triple[0]
            predicate = triple[1]
            object = triple[2]

            subject_tokenized = self.tokenizer.encode(subject)[:-1]
            predicate_tokenized = self.tokenizer.encode(predicate)[:-1]
            object_tokenized = self.tokenizer.encode(object)[:-1]

            input_ids.extend(subject_tokenized)
            attention_mask.extend([1] * len(subject_tokenized))
            role_ids.extend([1] * len(subject_tokenized))

            input_ids.extend(predicate_tokenized)
            attention_mask.extend([1] * len(predicate_tokenized))
            role_ids.extend([2] * len(predicate_tokenized))

            input_ids.extend(object_tokenized)
            attention_mask.extend([1] * len(object_tokenized))
            role_ids.extend([3] * len(object_tokenized))

            fact_str += subject + predicate + object
        
        input_ids.extend([self.tokenizer.eos_token_id])
        input_ids = self.pad_and_truncate(input_ids)
        attention_mask = self.pad_and_truncate(attention_mask)
        role_ids = self.pad_and_truncate(role_ids)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'role_ids': role_ids, 'fact_str': fact_str}
    
    def pad_and_truncate(self, ids):
        if len(ids) > self.max_source_length:
            return torch.tensor(ids[:self.max_source_length])
        else:
            return torch.tensor(ids + [self.tokenizer.pad_token_id] * (self.max_source_length - len(ids)))
            # above line is hacky, i.e. padding with pad_token_id, but it works for now. for attention_mask, it should be 0.

    @staticmethod
    def create_tokenizer(tokenizer_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        tokenizer.add_tokens(['S|', 'P|', 'O|'])
        print("we added S|, P|, O| to the tokenizer")
        return tokenizer

ds = DS('en_order.csv', tokenizer, 384, 128)
dl = DataLoader(ds, batch_size=4, num_workers=0,shuffle=False, collate_fn=collate)

from numpy.linalg import norm
from numpy import dot

BATCH_SIZE = 32
from tqdm import tqdm

ref_order = []
pred_order = []

# data_set =JsonL(jsonl_data,tokenizer,518)
# load_params = {"batch_size": BATCH_SIZE, "shuffle": False, "num_workers": 0}
# data_loader = DataLoader(data_set, **load_params, collate_fn = collate)


for _, data in enumerate(dl, 0):
#     print (data)
    
    sent = data['sent']
    fact_str = data['fact_str']
    lang = data['lang']
    ref_order += data['ref_order']

    # print(data)
    # break
    
    # print ("sent ",len(sent))
    # print ("fact ",len(fact_str))

# #     print (sent)
#     encoded_sentence = tokenizer.batch_encode_plus(sent, return_tensors="pt",padding=True)
#     encoded_facts = tokenizer.batch_encode_plus(fact_str, return_tensors="pt",padding=True)
    
    model.to(device)
    print("model shifted to device")
    with torch.no_grad():
        output = model(
            input_ids=torch.stack(data["input_ids"]).to(device),
            attention_mask=torch.stack(data["attention_mask"]).to(device),
            role_ids=torch.stack(data["role_ids"]).to(device),
            labels=torch.stack(data["labels"]).to(device),
            output_attentions=True  
        )
    
#     print (output.cross_attentions)
    print (output.cross_attentions[-1].shape)
    batch_fin_crossattention = output.cross_attentions[-1].mean(dim=1)
#     batch_fin_crossattention = output.cross_attentions[-1][:,2,:,:]
    print (batch_fin_crossattention.shape)
    print ("batch ",len(batch_fin_crossattention))
    
    TOK = ''
    for itr in range(len(batch_fin_crossattention)):
        
        fin_crossattention = batch_fin_crossattention[itr]
        print (fin_crossattention.shape)
    
        stflag = -1
        currfact = -1
        factpos = []
        factpos2 = []
        
        tokenized_sent = tokenizer.tokenize(sent[itr])
        tokenized_fact = tokenizer.tokenize(fact_str[itr])
        sent_len = len(tokenized_sent)
        role_ids = data['role_ids'][itr].tolist()
        for wordval in range(len(tokenized_fact)):

            tokenized_curr = tokenized_fact[wordval]
            try:
                tokenized_next = tokenized_fact[wordval+1]      
                if tokenized_next==tokenizer.tokenize(' ')[0]:
                    tokenized_next=tokenized_fact[wordval+2]
            except:
                tokenized_next=tokenized_curr
            # print(tokenized_next)
            if tokenized_curr != 'S|' and role_ids[wordval] == 1:
                continue
            
            if (tokenized_curr) == "S|" and stflag == -1:
                stflag = 1

                sumval = 0 
                fcount = 0
                maxpos = 0
                maxscore = -99
                print ("FIRST FACT")
                TOK = tokenized_next
                continue

            elif (tokenized_curr) == "S|" and stflag != -1 and tokenized_next==TOK:
                factpos.append(maxpos)
                factpos2.append(sumval/fcount)

                stflag = 1

                sumval = 0 
                fcount = 0
                maxpos = 0
                maxscore = -99
                print (len(factpos) , " FACT OVER")

                continue 


            elif (tokenized_curr) == "<QR>":
    #             stflag = 0
                print ("QR FOUNDS")
                continue

            elif (tokenized_curr) == "<S>":
                
                continue


#             print(fin_crossattention[:,wordval][:sent_len])
            max_fact = fin_crossattention[:,wordval][:sent_len].argmax()
            argval = fin_crossattention[:,wordval][:sent_len].max()
            try:
                # pass
                print ("fact subword : ",tokenized_fact[wordval],end = "\t\t")
            except :
                print ("hello")
                pass
            
            intval = np.array(max_fact.to('cpu'))
            # tempind = encoded_facts.word_ids()[intval]
            print ("matched sentence sub word : ",tokenized_sent[intval])

            if stflag == 1:
                sumval += (intval+1)
                fcount += 1
                if argval > maxscore:
                    maxpos = (intval+1)
                    maxscore = argval

        factpos.append(maxpos)
        factpos2.append(sumval/fcount)
        stflag = 0
        print (len(factpos) , "LAST FACT OVER")
        print ("===end===")
        final_ordering =  np.argsort(factpos)
        pred_order.append(list(final_ordering))
        # pred_order.append(list(range(len(list(final_ordering)))))
        # pred_order.append(list(range(len(final_ordering)-1, -1, -1)))
        
        #following 3 lines for random shuffle
        # temp_list = list(final_ordering)
        # random.shuffle(temp_list)
        # pred_order.append(temp_list)
        
        if len(list(final_ordering)) != len(data['ref_order'][itr]):
            print(data['ref_order'][itr])
            print('***')
            print(pred_order)
            print('***')
            print(data['fact_str'][itr])
            print('***')
            print(tokenized_fact)

        print ()
        print ("works")

def get_accuracy(a, b):
    assert len(a)==len(b), "unequal size of list"
    count = 0
    for y, z in zip(a, b):
        if y==z:
            count+=1
    return count/len(a)

from scipy.stats import kendalltau, spearmanr
import numpy as np

avg_spearman = np.array([spearmanr(y, z)[0] for y, z in zip(pred_order, ref_order)])
avg_kendalltau = np.array([kendalltau(y, z)[0] for y, z in zip(pred_order, ref_order)])

print("spearman", avg_spearman.mean())
print("kendalltau", avg_kendalltau.mean())
print("accuracy", get_accuracy(ref_order, pred_order))