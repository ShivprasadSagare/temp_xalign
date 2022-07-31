import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM
import torch
from sacrebleu import BLEU
from indicnlp.transliterate import unicode_transliterate
import torch.nn as nn
import numpy as np
from typing import Optional, Union
import pandas as pd

from transformers import (
    BeamSearchScorer,
    BeamScorer,
    LogitsProcessorList, 
    StoppingCriteriaList,
    MaxLengthCriteria
)
from transformers.generation_utils import BeamSearchOutput

class FineTuner(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams.model_name_or_path
        )
        self.model.resize_token_embeddings(len(self.hparams.tokenizer))
        self.gate = nn.Linear(self.model.config.d_model, 1)
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

    def get_attn_key_pad_mask(self, seq_k, seq_q):
        ''' For masking out the padding part of key sequence. '''
        # Expand to fit the shape of key query attention matrix.
        len_q = seq_q.size(1)
        padding_mask = seq_k.eq(self.hparams.tokenizer.pad_token_id)
        padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
        return padding_mask

    def forward(self, input_ids, attention_mask, role_ids, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, role_ids=role_ids, labels=labels, output_hidden_states=True)
        output_logits = outputs.logits.softmax(dim=-1)

        # copy mechanism code
        decoder_last_states = outputs.decoder_hidden_states[-1]
        encoder_last_states = outputs.encoder_last_hidden_state
        copy_gate = torch.sigmoid(self.gate(decoder_last_states))
        full_vocab_prob = (1 - copy_gate) * output_logits
        scores = torch.bmm(decoder_last_states, encoder_last_states.transpose(2, 1))
        dec_enc_attn_mask = self.get_attn_key_pad_mask(input_ids, labels)

        scores = scores.masked_fill(dec_enc_attn_mask, -np.inf)
        oov_vocab_prob = torch.softmax(scores, -1)
        full_vocab_prob = full_vocab_prob.scatter_add(2, input_ids.unsqueeze(1).repeat(1, full_vocab_prob.shape[1], 1), oov_vocab_prob * copy_gate)
        return torch.log(full_vocab_prob + 1e-8)

    def beam_search(
        self,
        encoder_input_ids: torch.LongTensor, 
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs,
        ) -> Union[BeamSearchOutput, torch.LongTensor]:
            # init values
            logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
            stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
            pad_token_id = pad_token_id if pad_token_id is not None else self.model.config.pad_token_id
            eos_token_id = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id

            # init attention / hidden states / scores tuples
            scores = None
            decoder_attentions = None
            cross_attentions = None
            decoder_hidden_states = None

            batch_size = len(beam_scorer._beam_hyps)
            num_beams = beam_scorer.num_beams

            batch_beam_size, cur_len = input_ids.shape

            if num_beams * batch_size != batch_beam_size:
                raise ValueError(
                    f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
                )

            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
            beam_scores[:, 1:] = -1e9
            beam_scores = beam_scores.view((batch_size * num_beams,))

            while True:
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                    output_hidden_states=True
                )

                output_logits = outputs.logits.softmax(dim=-1)
                decoder_last_states = outputs.decoder_hidden_states[-1]
                encoder_last_states = outputs.encoder_last_hidden_state
                copy_gate = torch.sigmoid(self.gate(decoder_last_states))
                full_vocab_prob = (1 - copy_gate) * output_logits
                scores = torch.bmm(decoder_last_states, encoder_last_states.transpose(2, 1))
    #             print("encoder_input_ids", encoder_input_ids.shape)
    #             print("decoder_input_ids", input_ids.shape)
                dec_enc_attn_mask = self.get_attn_key_pad_mask(encoder_input_ids, input_ids)

                scores = scores.masked_fill(dec_enc_attn_mask, -np.inf)
                oov_vocab_prob = torch.softmax(scores, -1)
                full_vocab_prob = full_vocab_prob.scatter_add(2, encoder_input_ids.unsqueeze(1).repeat(1, full_vocab_prob.shape[1], 1), oov_vocab_prob * copy_gate)
                final_output_logits = torch.log(full_vocab_prob + 1e-8)

                next_token_scores = final_output_logits[:, -1, :]

                next_token_scores = logits_processor(input_ids, next_token_scores)
                next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

                # reshape for beam search
                vocab_size = next_token_scores.shape[-1]
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )

                next_indices = (next_tokens / vocab_size).long()
                next_tokens = next_tokens % vocab_size

                # stateless
                beam_outputs = beam_scorer.process(
                    input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
                )
                if model_kwargs["past"] is not None:
                    model_kwargs["past"] = self.model._reorder_cache(model_kwargs["past"], beam_idx)

                # increase cur_len
                cur_len = cur_len + 1

                if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                    break

            sequence_outputs = beam_scorer.finalize(
                input_ids,
                beam_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                max_length=stopping_criteria.max_length,
            )

            return sequence_outputs["sequences"]

    def custom_generate(self,
                    inputs, 
                    attention_mask, 
                    num_beams=5,
                    max_length=200,
                    length_penalty=1.0, 
                    **model_kwargs):
        pad_token_id = self.model.config.pad_token_id
        bos_token_id = self.model.config.bos_token_id
        eos_token_id = self.model.config.eos_token_id

        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])
        logits_processor = LogitsProcessorList()

        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(inputs, bos_token_id, model_kwargs)

        batch_size = inputs_tensor.shape[0]
        model_kwargs.update({"attention_mask": attention_mask, "use_cache": True})

        model_kwargs = self.model._prepare_encoder_decoder_kwargs_for_generation(
                            inputs_tensor, model_kwargs, model_input_name
                        )

        # decoder input ids
        input_ids = self.model._prepare_decoder_input_ids_for_generation(
                    batch_size,
                    decoder_start_token_id=None,
                    bos_token_id=bos_token_id,
                    model_kwargs=model_kwargs,
                )

        beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    num_beams=num_beams,
                    device=inputs_tensor.device,
                    length_penalty=length_penalty,
                    do_early_stopping=False,
                )

        input_ids, model_kwargs = self.model._expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=True, **model_kwargs)

        expanded_return_idx = (
                torch.arange(inputs_tensor.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(inputs_tensor.device)
            )

        encoder_input_ids = inputs_tensor.index_select(0, expanded_return_idx)

        return self.beam_search(
                    encoder_input_ids,
                    input_ids,
                    beam_scorer,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    **model_kwargs,
                )

    def _step(self, batch):
        input_ids, attention_mask, role_ids, labels = batch['input_ids'], batch['attention_mask'], batch['role_ids'], batch['labels']
        lm_labels = torch.clone(labels)
        lm_labels[lm_labels[:, :] == self.hparams.tokenizer.pad_token_id] = -100
        
        logits = self(input_ids, attention_mask, role_ids, labels)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), lm_labels.view(-1))
        return loss

    def _generative_step(self, batch):
        generated_ids = self.custom_generate(
            batch['input_ids'],
            batch['attention_mask'],
            role_ids=batch['role_ids'],
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

        random_indices = set([len(input_text)//i for i in range(2, 7)])
        epoch_list = [self.trainer.current_epoch for i in range(len(random_indices))]

        input_text = [input_text[i] for i in random_indices]
        pred_text = [pred_text[i] for i in random_indices]
        ref_text = [ref_text[i] for i in random_indices]

        data = [i for i in zip(epoch_list, input_text, ref_text, pred_text)]
        self.trainer.logger.log_text(key='validation_predictions', data=data, columns=['epoch', 'input_text', 'ref_text', 'pred_text'])

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

        # random_indices = set([len(self.languages_map['hi']['original_input_text'])//i for i in range(2, 7)])
        # epoch_list = [self.trainer.current_epoch for i in range(len(random_indices))]

        # input_text = [self.languages_map['hi']['original_input_text'][i] for i in random_indices]
        # pred_text = [self.languages_map['hi']['original_pred_text'][i] for i in random_indices]
        # ref_text = [self.languages_map['hi']['original_ref_text'][i] for i in random_indices]

        # data = [i for i in zip(epoch_list, input_text, ref_text, pred_text)]
        # self.trainer.logger.log_text(key='validation_predictions', data=data, columns=['epoch', 'input_text', 'ref_text', 'pred_text'])

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
        
        df_to_write.to_csv('predictions_transliterated.csv', sep='\t')

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