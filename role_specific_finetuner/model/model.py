import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM
import torch
from sacrebleu import BLEU
from indicnlp.transliterate import unicode_transliterate
import pandas as pd


class FineTuner(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams.model_name_or_path
        )
        self.model.resize_token_embeddings(len(self.hparams.tokenizer))
        self.cal_bleu = BLEU()
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
        self.lang_id_map = {v['id']: k for k, v in self.languages_map.items()}

    def forward(self, input_ids, attention_mask, role_ids, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, role_ids=role_ids, labels=labels)
        return outputs

    def _step(self, batch):
        input_ids, attention_mask, role_ids, labels = batch['input_ids'], batch['attention_mask'], batch['role_ids'], batch['labels']
        outputs = self(input_ids, attention_mask, role_ids, labels)
        loss = outputs[0]
        return loss

    def _generative_step(self, batch):
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            role_ids=batch['role_ids'],
            use_cache=True,
            num_beams=self.hparams.eval_beams,
            max_length=self.hparams.tgt_max_seq_len
            # understand above 3 arguments
            )

        input_text = self.hparams.tokenizer.batch_decode(
            batch['input_ids'],
            skip_special_tokens=True)
        pred_text = self.hparams.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        batch['labels'][batch['labels'] == -100] = self.hparams.tokenizer.pad_token_id
        ref_text = self.hparams.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        return input_text, pred_text, ref_text

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, on_epoch=True)
        if batch_idx==0:
            input_text, pred_text, ref_text = self._generative_step(batch)
            return {'val_loss': loss, 'input_text': input_text, 'pred_text': pred_text, 'ref_text': ref_text}

    def validation_epoch_end(self, outputs):
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # self.log("epoch_val_loss", avg_loss)

        input_text = []
        pred_text = []
        ref_text = []
        for x in outputs:
            input_text.extend(x['input_text'])
            pred_text.extend(x['pred_text'])
            ref_text.extend(x['ref_text'])
        bleu = self.cal_bleu.corpus_score(pred_text, [ref_text])    
        self.log("val_bleu", bleu.score)

        # random_indices = set([len(input_text)//i for i in range(2, 7)])
        epoch_list = [self.trainer.current_epoch for i in range(len(input_text))]

        # input_text = [input_text[i] for i in random_indices]
        # pred_text = [pred_text[i] for i in random_indices]
        # ref_text = [ref_text[i] for i in random_indices]

        data = [i for i in zip(epoch_list, input_text, ref_text, pred_text)]
        self.trainer.logger.log_text(key='validation_predictions', data=data, columns=['epoch', 'input_text', 'ref_text', 'pred_text'])
        return  

    def test_step(self, batch, batch_idx):
        input_text, pred_text, ref_text = self._generative_step(batch)
        lang = batch['lang'].tolist()
        return {'input_text': input_text, 'pred_text': pred_text, 'ref_text': ref_text, 'lang': lang}

    # converting from unified devnagari script to original lang script
    def get_native_text_from_unified_script(self, unified_text, lang):
        return unicode_transliterate.UnicodeIndicTransliterator.transliterate(unified_text, "hi", lang)

    def process_for_bleu(self, text, lang):
        native_text = text
        if lang!='en':
            # convert unified script to native langauge text
            native_text = self.get_native_text_from_unified_script(text, lang)

        native_text = native_text.strip()
        # as input and predicted text are already space tokenized
        native_text = ' '.join([x for x in native_text.split()])
        return native_text

    def test_epoch_end(self, outputs):
        df_to_write = pd.DataFrame(columns=['lang', 'input_text', 'ref_text', 'pred_text', 'bleu'])
        input_texts = []
        pred_texts = []
        ref_texts = []
        langs = []
        for x in outputs:
            input_texts.extend(x['input_text'])
            pred_texts.extend(x['pred_text'])
            ref_texts.extend(x['ref_text'])
            langs.extend(x['lang'])
        
        for key in self.languages_map:
            self.languages_map[key]['original_pred_text'] = [self.process_for_bleu(pred_text, self.lang_id_map[lang]) for pred_text, lang in zip(pred_texts, langs) if lang == self.languages_map[key]['id']]
            self.languages_map[key]['original_ref_text'] = [self.process_for_bleu(ref_text, self.lang_id_map[lang]) for ref_text, lang in zip(ref_texts, langs) if lang == self.languages_map[key]['id']]        
            self.languages_map[key]['original_input_text'] = [self.process_for_bleu(input_text, self.lang_id_map[lang]) for input_text, lang in zip(input_texts, langs) if lang == self.languages_map[key]['id']]

        # for pred_text, ref_text, lang in zip(pred_texts, ref_texts, langs):
        #     self.languages_map[self.lang_id_map[lang]]['pred_text'].append(pred_text)

        overall_bleu = 0
        for key in self.languages_map:
        # for key in ['en']:
            try:
                self.languages_map[key]['bleu'] = self.cal_bleu.corpus_score(self.languages_map[key]['original_pred_text'], [self.languages_map[key]['original_ref_text']]).score
                self.log(f"test_bleu_{key}", self.languages_map[key]['bleu'])
                overall_bleu += self.languages_map[key]['bleu']
            except:
                pass

        self.log("test_bleu", overall_bleu/len(self.languages_map))

        for key in self.languages_map:
            l = len(self.languages_map[key]['original_pred_text'])
            self.languages_map[key]['bleus'] = [self.cal_bleu.corpus_score([self.languages_map[key]['original_pred_text'][i]], [[self.languages_map[key]['original_ref_text'][i]]]).score for i in range(len(self.languages_map[key]['original_pred_text']))]
            df_key = pd.DataFrame({
                'lang':[key for i in range(l)],
                'input_text':[self.languages_map[key]['original_input_text'][i] for i in range(l)],
                'pred_text':[self.languages_map[key]['original_pred_text'][i] for i in range(l)],
                'ref_text':[self.languages_map[key]['original_ref_text'][i] for i in range(l)],
                'bleu':[self.languages_map[key]['bleus'][i] for i in range(l)]
            })
            df_to_write = pd.concat([df_to_write, df_key])
        run_name = self.hparams['run_name']
        df_to_write.to_csv(f'predictions_{run_name}.csv', sep='\t')
        # random_indices = set([len(self.languages_map['hi']['original_input_text'])//i for i in range(2, 7)])
        # epoch_list = [self.trainer.current_epoch for i in range(len(random_indices))]

        # input_text = [self.languages_map['hi']['original_input_text'][i] for i in random_indices]
        # pred_text = [self.languages_map['hi']['original_pred_text'][i] for i in random_indices]
        # ref_text = [self.languages_map['hi']['original_ref_text'][i] for i in random_indices]

        # data = [i for i in zip(epoch_list, input_text, ref_text, pred_text)]
        # self.trainer.logger.log_text(key='validation_predictions', data=data, columns=['epoch', 'input_text', 'ref_text', 'pred_text'])

        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Model args')
        parser.add_argument('--learning_rate', default=2e-5, type=float)
        parser.add_argument('--model_name_or_path', default='t5-base', type=str)
        parser.add_argument('--eval_beams', default=4, type=int)
        parser.add_argument('--tgt_max_seq_len', default=128, type=int)
        return parent_parser