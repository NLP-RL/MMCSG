import os
import numpy as np
import pandas as pd
import json
import warnings
import logging
import gc
import random
import math
import pickle
import re
import ast
import torch
from tqdm import tqdm
from typing import Optional
from datetime import datetime
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.translate.bleu_score import sentence_bleu
from torch import nn
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from evaluate import load
from transformers import AutoTokenizer, AutoModel
bertscore = load("bertscore")

from rouge_score.rouge_scorer import RougeScorer
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
#from torchmetrics.text.bert import BERTScore
from torchmetrics.functional.text.bert import bert_score
import random
from sentence_transformers import SentenceTransformer
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings


Sem_model = SentenceTransformer('bert-base-nli-mean-tokens')

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(42)



warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

print(torch.cuda.is_available())
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    BartDecoder,
    BartLearnedPositionalEmbedding,
    BartEncoderLayer,
    shift_tokens_right,
    _make_causal_mask,
    _expand_mask
)

from transformers.models.bart.configuration_bart import BartConfig

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from transformers.modeling_utils import PreTrainedModel, unwrap_model

from transformers import (
    BartTokenizerFast,
    AdamW
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput
)


from transformer_encoder import TransformerEncoder

SOURCE_COLUMN = 'concated_transcript'
TARGET_COLUMN = "mmcs"
VISUAL_INPUT_PATH = 'Image_features.json'


SOURCE_MAX_LEN = 480
TARGET_MAX_LEN = 50
MAX_UTTERANCES = 25

VISUAL_DIM = 2048
VISUAL_MAX_LEN = 1 

ACOUSTIC_DIM = 1042
ACOUSTIC_MAX_LEN = 1


# KG_DIM = 768
# KG_MAX_LEN = 1

PPCC_DIM = 768
PPCC_MAX_LEN = 14


BATCH_SIZE = 16
MAX_EPOCHS = 60

BASE_LEARNING_RATE = 5e-6
NEW_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

NUM_BEAMS = 5
EARLY_STOPPING = True
CLS_LR = 2e-3
NO_REPEAT_NGRAM_SIZE = 3

Encoder_Cls_dim = 768
Num_labels = 3
EARLY_STOPPING_THRESHOLD = 5
intent_column = "Intent"

MODEL_OUTPUT_DIR = '/home/jupyter/MMCSG/MMCS/'
RESULT_OUTPUT_DIR = '/home/jupyter/MMCSG/MMCS/'

def read_json_data(path):
    f = open(path)
    data = json.load(f)
    f.close()
    del f
    gc.collect()
    return data

def preprocess_dataset(dataset):

    source = list(dataset[SOURCE_COLUMN].values)
    # print(type(source))
    model_inputs = TOKENIZER(source,
                                    max_length=SOURCE_MAX_LEN,
                                    padding='max_length',
                                    truncation=True)

    target = list(dataset[TARGET_COLUMN].values)
    with TOKENIZER.as_target_tokenizer():
        labels = TOKENIZER(target,
                            max_length=TARGET_MAX_LEN,
                            padding='max_length',
                            truncation=True)  
       
        # IMP:
        # Replace all tokenizer.pad_token_id in the labels by -100 to ignore padding tokens in the loss.
        labels['input_ids'] = [[(l if l != TOKENIZER.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

    model_inputs['input_ids'] = torch.tensor([i for i in model_inputs['input_ids']], dtype=torch.long, device=DEVICE)
    model_inputs['attention_mask'] = torch.tensor([a for a in model_inputs['attention_mask']], dtype=torch.long, device=DEVICE)
    model_inputs['visual_input'] = torch.tensor(dataset["image_features"]).to(DEVICE)
    model_inputs['audio_input'] = torch.tensor(dataset["audio_features"]).to(DEVICE)
    model_inputs['ppc_input'] = torch.tensor(dataset["personality_features"]).to(DEVICE)
    model_inputs['intent_labels'] = torch.tensor(dataset["encoded_merged_intent"], dtype=torch.long).to(DEVICE) 
    model_inputs['labels'] = torch.tensor([l for l in labels['input_ids']], dtype=torch.long, device=DEVICE)

    del target
    del labels
    gc.collect()
    return model_inputs


def set_up_data_loader(dataset):
    print("Setup data loader")
    dataset = preprocess_dataset(dataset)
    dataset = TensorDataset(dataset['input_ids'],
                            dataset['attention_mask'],
                            dataset['visual_input'],
                            dataset['audio_input'],
                            dataset['ppc_input'],
                            dataset['labels'],
                            dataset['intent_labels']
                            )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
# ---------------------------------------------- Multimodal Context Aware Attention ----------------------------------------------

class ContextAwareAttention(nn.Module):

    def __init__(self,
                 dim_model: int,
                 dim_context: int,
                 dropout_rate: Optional[float]=0.0):
        super(ContextAwareAttention, self).__init__()
        
        self.dim_model = dim_model
        self.dim_context = dim_context
        self.dropout_rate = dropout_rate
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.dim_model, 
                                                     num_heads=1, 
                                                     dropout=self.dropout_rate, 
                                                     bias=True,
                                                     add_zero_attn=False,
                                                     batch_first=True,
                                                     device=DEVICE)


        self.u_k = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_k = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_k = nn.Linear(self.dim_model, 1, bias=False)
        
        self.u_v = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_v = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_v = nn.Linear(self.dim_model, 1, bias=False)
        




    def forward(self,
                q: torch.Tensor, 
                k: torch.Tensor,
                v: torch.Tensor,
                context: Optional[torch.Tensor]=None):
        
        key_context = self.u_k(context)
        value_context = self.u_v(context)

        lambda_k = F.sigmoid(self.w1_k(k) + self.w2_k(key_context))
        lambda_v = F.sigmoid(self.w1_v(v) + self.w2_v(value_context))

        k_cap = (1 - lambda_k) * k + lambda_k * key_context
        v_cap = (1 - lambda_v) * v + lambda_v * value_context

        attention_output, _ = self.attention_layer(query=q,
                                                   key=k_cap,
                                                   value=v_cap)
        return attention_output
         




class MultimodalBartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = MultimodalBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.classifier = nn.Sequential(nn.Linear(Encoder_Cls_dim,Num_labels),
                                        nn.Softmax(-1))
        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        visual_input=None,      # New addition of visual_input
        audio_input = None,
        ppc_input=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        test = True
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            visual_input=visual_input,      # New addition of visual_input
            audio_input = audio_input,
            ppc_input =ppc_input,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
           
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = 0
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
#         loss_cls = CrossEntropyLoss()
       
        # classification_loss = 0
        # if intent_labels is not None:
        #     print(outputs.keys())
        #     # print(outputs.encoder_last_hidden_state.shape)
        #     cls_logits = self.classifier(outputs.encoder_last_hidden_state.mean(dim = 1)) # New addition
        #     # print(cls_logits.shape,intent_labels.shape)
        #     classification_loss = loss_cls(cls_logits.view(-1,Num_labels),intent_labels.view(-1))
        #     Total_loss = classification_loss #+ masked_lm_loss
        cls_logits = self.classifier(outputs.encoder_last_hidden_state.mean(dim = 1)) # New addition



        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
       
       
        if test :
            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
        else:
            return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            # cls_logits = cls_logits, # New addition
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        ) , cls_logits


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

class MultimodalBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = MultimodalBartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        visual_input=None,      # New addition of visual_input
        audio_input = None,
        ppc_input=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_input=visual_input,      # New addition of visual_input
                audio_input = audio_input,
                ppc_input = ppc_input,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

# ---------------------------------------------- Modality Aware Fusion ----------------------------------------------

class MAF_I(nn.Module):
    
    def __init__(self,
                 dim_model: int,
                 dropout_rate: int):
        super(MAF_I, self).__init__()
        self.dropout_rate = dropout_rate
        
        self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias=False)


        self.visual_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                              dim_context=VISUAL_DIM,
                                                              dropout_rate=dropout_rate)   

  

        self.visual_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

        
        
        
        
    def forward(self,
                text_input: torch.Tensor,
                visual_context: Optional[torch.Tensor]=None,
                audio_context: Optional[torch.Tensor]=None,
                ppc_context: Optional[torch.Tensor]=None,):
                    
 
        # Video as Context for Attention
        visual_context = visual_context.permute(0, 2, 1)
        visual_context = self.visual_context_transform(visual_context)
        visual_context = visual_context.permute(0, 2, 1)
        
        video_out = self.visual_context_attention(q=text_input,
                                                  k=text_input,
                                                  v=text_input,
                                                  context=visual_context)


     
        

        
        # Global Information Fusion Mechanism
        weight_v = F.sigmoid(self.visual_gate(torch.cat((video_out, text_input), dim=-1)))

        output = self.final_layer_norm(text_input +
                                       weight_v * video_out)


        return output


class MAF_A(nn.Module):
    
    def __init__(self,
                 dim_model: int,
                 dropout_rate: int):
        super(MAF_A, self).__init__()
        self.dropout_rate = dropout_rate
        
    

        self.audio_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias=False)


        self.audio_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                              dim_context=ACOUSTIC_DIM,
                                                              dropout_rate=dropout_rate)  
        self.audio_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

        
        
        
        
    def forward(self,
                text_input: torch.Tensor,
                visual_context: Optional[torch.Tensor]=None,
                audio_context: Optional[torch.Tensor]=None,
                ppc_context: Optional[torch.Tensor]=None,):
                    
 
        # Video as Context for Attention
       
         # kg as Context for Attention
        audio_context = audio_context.permute(0, 2, 1)
        audio_context = self.audio_context_transform(audio_context)
        audio_context = audio_context.permute(0, 2, 1)
        
        audio_out = self.audio_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=audio_context)


        

        
        # Global Information Fusion Mechanism
        weight_a = F.sigmoid(self.audio_gate(torch.cat((audio_out, text_input), dim=-1)))
        
        output = self.final_layer_norm(text_input + weight_a * audio_out)


        return output
    


class MAF_P(nn.Module):
    
    def __init__(self,
                 dim_model: int,
                 dropout_rate: int):
        super(MAF_P, self).__init__()
        self.dropout_rate = dropout_rate
        
      

        self.ppc_context_transform = nn.Linear(PPCC_MAX_LEN, SOURCE_MAX_LEN, bias=False)


        self.ppc_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                              dim_context=PPCC_DIM,
                                                              dropout_rate=dropout_rate)   

       
        self.ppc_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

        
        
        
        
    def forward(self,
                text_input: torch.Tensor,
                visual_context: Optional[torch.Tensor]=None,
                audio_context: Optional[torch.Tensor]=None,
                ppc_context: Optional[torch.Tensor]=None,):
                    
 
      


        ppc_context = ppc_context.permute(0, 2, 1)
        ppc_context = self.ppc_context_transform(ppc_context)
        ppc_context = ppc_context.permute(0, 2, 1)
        
        ppc_out = self.ppc_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=ppc_context)
        

        
        # Global Information Fusion Mechanism
        weight_ppc = F.sigmoid(self.ppc_gate(torch.cat((ppc_out, text_input), dim=-1)))

        output = self.final_layer_norm(text_input + weight_ppc * ppc_out)


        return output
       

# ---------------------------------------------- Multimodal BartEncoder ----------------------------------------------

class MultimodalBartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()
        self.gradient_checkpointing = False

        # ================================ Modifications ================================ #
        self.fusionI_at_layer = [5]
        self.fusionA_at_layer = [5]
        self.fusionP_at_layer = [4]
        self.visual_transformer = TransformerEncoder(d_model=VISUAL_DIM, 
                                                     num_layers=4,
                                                     num_heads=8, 
                                                     dim_feedforward=VISUAL_DIM)
        self.audio_transformer = TransformerEncoder(d_model=ACOUSTIC_DIM, 
                                                       num_layers=4,
                                                       num_heads=2, 
                                                       dim_feedforward=ACOUSTIC_DIM)

        self.ppc_transformer = TransformerEncoder(d_model=PPCC_DIM, 
                                                       num_layers=4,
                                                       num_heads=8, 
                                                       dim_feedforward=PPCC_DIM)
        # =============================================================================== #

        self.MAF_layerI = MAF_I(dim_model=embed_dim, dropout_rate=0.2)
        self.MAF_layerA = MAF_A(dim_model=embed_dim, dropout_rate=0.2)
        self.MAF_layerP = MAF_P(dim_model=embed_dim, dropout_rate=0.2)
       

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        visual_input=None,      # New addition of visual_input
        audio_input=None,
        ppc_input=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
           
             # ================================ Modifications ================================ #
            if idx in self.fusionI_at_layer:
                visual_input = self.visual_transformer(visual_input)[-1]
                hidden_states = self.MAF_layerI(text_input=hidden_states,
                                               visual_context=visual_input)
                

            if idx in self.fusionA_at_layer:
                audio_input = self.audio_transformer(audio_input)[-1]
                hidden_states = self.MAF_layerA(text_input=hidden_states,
                                               audio_context=audio_input)
            
            if idx in self.fusionP_at_layer:
                ppc_input = self.ppc_transformer(ppc_input)[-1]
                hidden_states = self.MAF_layerP(text_input=hidden_states,
                                               ppc_context = ppc_input)
                

            # =============================================================================== #
              
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                                 

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

def _save(model, 
          output_dir: str,
          tokenizer=None,
          state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        os.makedirs(output_dir, exist_ok=True)
        #print(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model, PreTrainedModel):
            if isinstance(unwrap_model(model), PreTrainedModel):
                if state_dict is None:
                    state_dict = model.state_dict()
                unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(output_dir, state_dict=state_dict)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
#         torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def save_model(model, 
               output_dir: str,
               tokenizer=None, 
               state_dict=None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the main process.
        """
        _save(model,output_dir, tokenizer=tokenizer, state_dict=state_dict)


def val_epoch(model,
              data_loader,
              optimizer):
    model.eval()
    epoch_val_loss = 0.0
    actuals = []
    predictions = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Validation Loss Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, visual_input, audio_input, ppc_input, labels, intent_labels = batch
            visual_input = visual_input.unsqueeze(dim = 1)
            audio_input = audio_input.unsqueeze(dim = 1)
            ppc_input = ppc_input.squeeze(dim = 1)

            outputs, logits = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            visual_input=visual_input,
                            audio_input=audio_input,
                            ppc_input=ppc_input,
                            labels=labels,
                            test = False
                            )
            loss = outputs['loss']
            pred = list(torch.argmax(logits,dim  = -1).cpu().detach().numpy())
            # act = list(intent_labels.cpu().detach().numpy())

            predictions.extend(pred)
            # actuals.extend(act)
            print(logits.shape)
            epoch_val_loss += loss.item()




    # accuracy = f1_score(actuals,predictions, average = "macro")
   


    del pred
    # del act
    del batch
    del input_ids
    del attention_mask
    # del acoustic_input
    del visual_input
    del labels
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    print(epoch_val_loss)
   
    return epoch_val_loss


def train(model,
          tokenizer,
          train_data_loader,
          val_data_loader,
          test_data_loader,
          base_learning_rate,
          new_learning_rate,
          weight_decay,
          **gen_kwargs):
   
    optimizer = prepare_for_training(model=model,
                                     base_learning_rate=base_learning_rate,
                                     new_learning_rate=new_learning_rate,
                                     weight_decay=weight_decay)
    print(optimizer)
   
    train_losses = []
    val_losses = []
    val_rouge_2 = []
    patience = 1
   
    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = train_epoch(model,
                                 train_data_loader,
                                 optimizer)
        train_losses.append(train_loss)
        print(train_losses)
       
        val_loss = val_epoch(model,
                             val_data_loader,
                             optimizer)
        val_losses.append(val_loss)

        val_results = get_val_scores(model,
                                     tokenizer,
                                     val_data_loader,
                                     desc="Validation Generation Iteration",
                                     epoch=epoch,
                                     **gen_kwargs)
        val_rouge_2.append(val_results['rouge_2'])
       
        test_results = get_val_scores(model,
                                      tokenizer,
                                      test_data_loader,
                                      desc="Test Generation Iteration",
                                      epoch=epoch,
                                      **gen_kwargs)
   
        print("Epoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_validation_loss: {}\ttrain_acc: {}".format(epoch+1, train_loss, val_loss, min(val_losses), train_acc))
       
        print("\nval_rouge_1: {}\tval_rouge_2: {}\tval_rouge_L: {}\tval_bleu_1: {}\tval_bleu_2: {}\tval_bleu_3: {}\tval_bleu_4: {}\tval_meteor: {}".format(
        val_results['rouge_1'], val_results['rouge_2'], val_results['rouge_L'], val_results['bleu_1'], val_results['bleu_2'], val_results['bleu_3'], val_results['bleu_4'], val_results['meteor']))
       
        print("\ntest_rouge_1: {}\ttest_rouge_2: {}\ttest_rouge_L: {}\ttest_bleu_1: {}\ttest_bleu_2: {}\ttest_bleu_3: {}\ttest_bleu_4: {}\ttest_meteor: {}\ttest_BS_1: {}\test_JS: {}".format(test_results['rouge_1'], test_results['rouge_2'], test_results['rouge_L'], test_results['bleu_1'], test_results['bleu_2'], test_results['bleu_3'], test_results['bleu_4'], test_results['meteor'],test_results['BS'], test_results['JS'] ))
       
        #if epoch == MAX_EPOCHS - 1:
        path = "./Model/Model_TI"
        save_model(model,path,tokenizer)
        print("Model saved at path: ", path)
       
        if val_results['rouge_2'] < max(val_rouge_2):
            patience = patience + 1          
            if patience == EARLY_STOPPING_THRESHOLD:
                break
               
        else:
            patience = 1

        del train_loss
        del val_loss
        #del path
        gc.collect()
        torch.cuda.empty_cache()
def get_val_scores(model,
                   tokenizer,
                   data_loader,
                   desc,
                   epoch,
                   **gen_kwargs):
    predictions, gold = test_epoch(model,
                                   tokenizer,
                                   data_loader,
                                   desc=desc,
                                   **gen_kwargs)
    result = get_scores(predictions, gold)
    
    # if "Validation" in desc and epoch == MAX_EPOCHS - 1:
    #     val_df = pd.DataFrame(list(zip(gold, predictions)), columns=['actual_explanation', 'predicted_explanation'])
    #     file_name = "/home/abhisek_1921cs16/New/MMCSG/val-IPC_mmcs" + str(epoch+1) + "_val_results.csv"
    #     val_df.to_csv(file_name, index=False) 
    #     print("Validation File saved")
        
    #elif "Test" in desc and epoch == MAX_EPOCHS - 1:
    # test_df = pd.DataFrame(list(zip(gold, predictions)), columns=['actual_explanation', 'predicted_explanation'])
    # file_name = "/home/abhisek_1921cs16/New/MMCSG/Multimodal_summary_5/Test_results_T_I_DS/test-T-I-DS_mmcs.csv"
    # test_df.to_csv(file_name, index=False)  
    # print("Test File saved")
    if "Test" in desc:
        test_df = pd.DataFrame(list(zip(gold, predictions)), columns=['actual', 'predicted'])
        file_name = "./Gen/" + str(epoch+1) + "_test_MDS_TI_results.csv"
        test_df.to_csv(file_name, index=False)    
    print("Test File saved")
    
    del predictions
    del gold
    gc.collect()
    torch.cuda.empty_cache() 
    
    return result 
def test_epoch(model,
               tokenizer,
               data_loader,
               desc,
               **gen_kwargs):
    model.eval()
    predictions = []
    cls_pred = []
    actuals=[]
    gold = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc=desc)):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, visual_input, audio_input, ppc_input, labels, intent_labels= batch
            visual_input = visual_input.unsqueeze(dim = 1)
            audio_input = audio_input.unsqueeze(dim = 1)
            ppc_input = ppc_input.squeeze(dim = 1)
            generated_ids = model.generate(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           visual_input=visual_input, 
                                           audio_input=audio_input,
                                           ppc_input=ppc_input,
                                           **gen_kwargs)
            
            outputs, logits = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            visual_input=visual_input,
                            audio_input=audio_input,
                            ppc_input=ppc_input,
                            labels=labels,
                            test = False
                            )
            
            print(logits.shape)               
            pred = list(torch.argmax(logits,dim  = -1).cpu().detach().numpy())
            #act = list(intent_labels.cpu().detach().numpy())


            cls_pred.extend(pred)
            #actuals.extend(act)
            generated_ids = generated_ids.detach().cpu().numpy()
            generated_ids = np.where(generated_ids != -100, generated_ids, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            labels = labels.detach().cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
            predictions.extend(decoded_preds)
            gold.extend(decoded_labels)

    #accuracy = f1_score(actuals,cls_pred, average = "macro")
    
    
    del batch
    del input_ids
    del attention_mask
    del visual_input
    del labels
    del generated_ids
    del decoded_preds
    del decoded_labels
    gc.collect()
    torch.cuda.empty_cache() 
    
    return predictions, gold

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def get_scores(reference_list: list,
               hypothesis_list: list):
    count=0
    met=0
    bleu_1=0
    bleu_2=0
    bleu_3=0
    bleu_4=0
    rouge1=0
    bs = 0
    J = 0
    rouge2=0
    rougel = 0
    weights_1 = (1./1.,)
    weights_2 = (1./2. , 1./2.)
    weights_3 = (1./3., 1./3., 1./3.)
    weights_4 = (1./4., 1./4., 1./4., 1./4.)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    for reference, hypothesis in list(zip(reference_list, hypothesis_list)):
        scores = rouge_scorer.score(reference, hypothesis)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougel += scores['rougeL'].fmeasure

        met += meteor_score([word_tokenize(reference)], word_tokenize(hypothesis))

        Ref_E = Sem_model.encode(reference)
        Hyp_E = Sem_model.encode(hypothesis)

        bs += cosine_similarity([Ref_E],[Hyp_E])

        # print('ref:',reference)
        # print('hyp:',hypothesis)
        # print('co sine:',bs)


        reference = reference.split()
        hypothesis = hypothesis.split()
        
        # results = bertscore.compute(predictions=hypothesis, references=reference)

        J +=  jaccard(reference, hypothesis)


        bleu_1 += sentence_bleu([reference], hypothesis, weights_1) 
        bleu_2 += sentence_bleu([reference], hypothesis, weights_2)
        bleu_3 += sentence_bleu([reference], hypothesis, weights_3)
        bleu_4 += sentence_bleu([reference], hypothesis, weights_4)
        count += 1

    return {
        "rouge_1": rouge1*100/count,
        "rouge_2": rouge2*100/count,
        "rouge_L": rougel*100/count,
        "bleu_1": bleu_1*100/count,
        "bleu_2": bleu_2*100/count,
        "bleu_3": bleu_3*100/count,
        "bleu_4": bleu_4*100/count,
        "meteor": met*100/count,
        "JS": J/count,
        "BS": bs/count
    }

       
def prepare_for_training(model,
                         base_learning_rate: float,
                         new_learning_rate: float,
                         weight_decay: float):
    base_params_list = []
    new_params_list = []
    for name, param in model.named_parameters():
        if "acoustic_transformer" in name or "visual_transformer" in name or "ppc_transformer" in name or "MAF_layer" in name:
            new_params_list.append(param)
        else:
            base_params_list.append(param)
           
    optimizer = AdamW(
        [
            {'params': base_params_list,'lr': base_learning_rate, 'weight_decay': weight_decay},
            {'params': new_params_list,'lr': new_learning_rate, 'weight_decay': weight_decay}            
        ],
        lr=base_learning_rate,
        weight_decay=weight_decay
    )
   
    del base_params_list
    del new_params_list
    gc.collect()
    torch.cuda.empty_cache()
   
    return optimizer


def train_epoch(model,
                data_loader,
                optimizer):
    model.train()
    epoch_train_loss = 0.0
    actuals = []
    predictions = []
    for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, attention_mask, visual_input, audio_input, ppc_input, labels, intent_labels = batch
        optimizer.zero_grad()
        visual_input = visual_input.unsqueeze(dim = 1)
        audio_input = audio_input.unsqueeze(dim = 1)
        ppc_input = ppc_input.squeeze(dim = 1)

        # print(acoustic_input.shape,ACOUSTIC_DIM)
        outputs , logits = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        visual_input=visual_input,
                        audio_input=audio_input,
                        ppc_input=ppc_input,
                        labels=labels,
                        test = False
                        )
        loss = outputs['loss']
        epoch_train_loss += loss.item()

        pred = list(torch.argmax(logits,dim  = -1).cpu().detach().numpy())
        #act = list(intent_labels.cpu().detach().numpy())


        predictions.extend(pred)
        # actuals.extend(act)
           
        loss.backward()
        optimizer.step()
   
    # accuracy = f1_score(actuals,predictions, average = "macro")
   
    del pred
    # del act
    del batch
    del input_ids
    del attention_mask
    del visual_input
    del labels
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache()
   
    return epoch_train_loss/ step , predictions[0]




# ------------------------------------------------------------ MAIN MODEL ------------------------------------------------------------ #


MODEL = MultimodalBartForConditionalGeneration.from_pretrained('facebook/bart-base')
print("Model loaded...\n")

MODEL.to(DEVICE)


Sem_model = SentenceTransformer('bert-base-nli-mean-tokens')
Sem_model.to(DEVICE)

# device_id = torch.cuda.device_count()
# if torch.cuda.device._count()>1:
#    MODEL = nn.DataParallel(MODEL)

# MODEL = MODEL.to("cuda:{}".format(DEVICE))

# # MODEL.to("cuda:{}".format(device_id))


# # print(torch.cuda.device_count())
# # DEVICE_1 = torch.device("cuda",[1,2])
# # print(DEVICE_1)
 # MODEL.to(DEVICE)

TOKENIZER = BartTokenizerFast.from_pretrained('facebook/bart-base')
print("Tokenizer loaded...\n")

SOURCE_PREFIX = ''
TARGET_PREFIX = ''

print('TARGET_COLUMN:',TARGET_COLUMN)
print('MODEL_OUTPUT_DIR',MODEL_OUTPUT_DIR)
print('RESULT_OUTPUT_DIR',RESULT_OUTPUT_DIR)
print('SOURCE_PREFIX',SOURCE_PREFIX)
print('TARGET_PREFIX',TARGET_PREFIX)


gc.collect()

pytorch_total_params = sum(p.numel() for p in MODEL.parameters())
print("Total parameters: ", pytorch_total_params)
pytorch_total_train_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
print("Total trainable parameters: ", pytorch_total_train_params)


NewDataset_path = 'Dataset/Final.json'
Dataset = pd.DataFrame(read_json_data(NewDataset_path))
print(Dataset.keys())
	
# print(type(Dataset['Img_features'].values))

train_data , test_data = train_test_split(Dataset,test_size = 0.2)
valid_data , test_data = train_test_split(test_data,test_size = 0.7)
# print(len(test_data))
# test_data.to_csv("/home/abhisek_1921cs16/New/MMCSG/Multimodal_summary_5/Test_results_T_I_DS/test_data.csv")
print("Train test split")
# ------------------------------ READ DATASET ------------------------------ #

train_dataset = set_up_data_loader(train_data)
print("\nTraining Data Loaded...")

val_dataset = set_up_data_loader(valid_data)
print("\nValidation Data Loaded...")

test_dataset = set_up_data_loader(test_data)
print("\nTest Data Loaded...")
print("All data loaded")

gc.collect()


# ------------------------------ TRAINING SETUP ------------------------------ #

gen_kwargs = {
    'num_beams': NUM_BEAMS,
    'max_length': TARGET_MAX_LEN, 
    'early_stopping': EARLY_STOPPING,
    'no_repeat_ngram_size': NO_REPEAT_NGRAM_SIZE
}

train(model=MODEL,
      tokenizer=TOKENIZER,
      train_data_loader=train_dataset,
      val_data_loader=val_dataset,
      test_data_loader=test_dataset,
      base_learning_rate=BASE_LEARNING_RATE,
      new_learning_rate=NEW_LEARNING_RATE,
      weight_decay=WEIGHT_DECAY,
      **gen_kwargs)

print("Model Trained!")

