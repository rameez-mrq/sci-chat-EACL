import random
#fixinf sys version
try:
    import sys # Just in case
    start = sys.version.index('|') # Do we have a modified sys.version?
    end = sys.version.index('|', start + 1)
    version_bak = sys.version # Backup modified sys.version
    sys.version = sys.version.replace(sys.version[start:end+1], '') # Make it legible for platform module
    import platform
    platform.python_implementation() # Ignore result, we just need cache populated
    platform._sys_version_cache[version_bak] = platform._sys_version_cache[sys.version] # Duplicate cache
    sys.version = version_bak # Restore modified version string
except ValueError: # Catch .index() method not finding a pipe
    pass
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    GPT2Tokenizer, GPT2LMHeadModel, AutoModelForSeq2SeqLM, 
    BlenderbotTokenizer, BlenderbotForConditionalGeneration,
    BitsAndBytesConfig
    # LlamaForCausalLM, LlamaTokenizer,

)
from typing import Dict, List, Any
from peft import PeftModel
from tqdm.auto import tqdm
from pathlib import Path
import torch
import requests
path = lambda p: Path(p).absolute().resolve()
device = "cuda:0"
# print(device)
# torch.multiprocessing.set_start_method('spawn')

class PersonaManager:
    def __init__(self):
        '''
        Please put "personas.txt" and "utils.py" in the same directory.
        You can also set `persona_file` as the absolute path to "degraded_random_responses_filtered.txt" file.
        '''
        persona_file = path(__file__).parent.joinpath("topics_podcast.txt")
        with open(persona_file) as f:
            self.all_personas = f.read().strip().splitlines()
        
    def get_persona(self, seed=1):
        # seed is randomly generated between 0~1000, where any seed > 500 means no persona
        if seed > 500:
            return []
        random.seed(int(seed))
        return random.sample(self.all_personas, k=5)
    
    def get_single_persona(self):
        random.seed()
        return random.choice(self.all_personas)


class DialogueModels:
    def __init__(self):
        modelclass_dict = {
            "qc": QcModel,
            # "scichat1": Model1,
            # "scichat2": Model2,
            # "scichat3": Model3,
            "vanilla_dialogpt": DialogGPT,
            "bart": PersonaChatBART,
            "dialogpt": PersonaChatGPT,
            "blenderbot": Blenderbot,
        }
        self.models = {}
        pbar = tqdm(modelclass_dict.items())
        for modelname,modelclass in pbar:
            pbar.set_description(f"Loading {modelclass.__name__}")
            self.models[modelname] = modelclass()
        
        self.modelnames = list(self.models.keys())

    def get_response(self,data):
        model = data['model']
        response = self.models[model].response(
            data['user_input'],
            data['history'],
            data['personas'],
            
        )
        print( "Model:", model, "Response:", response)
        return response

#SciChat Models
class Model3:
    def __init__(self):
        self.url = "API URL"
        self.headers = {
            "Accept": "application/json",
            "Authorization": "TOKEN",
            "Content-Type": "application/json"
        }
        
    def response(self,user_input,dial_history,personas):
        url = self.url
        headers = self.headers
        text = personas + ". " + user_input
        data = {
            "inputs": text,
            "parameters": {}
        }
        
        response = requests.post(url, json=data, headers=headers)
        reply = response.json()
        
        return reply[0]['generated_reply'][12:]

class Model1:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Model_name", padding_side='left')
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Model_name")
        self.model.to(device)
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
        self.bos = self.tokenizer.bos_token
    
    def preprocess_persona(self,personas):
        return " ".join(personas)
  
    def preprocess_dial_history(self,dial_history):
        return " EOS ".join(dial_history)
        
    def response(self,user_input,dial_history,personas):
        # print("Persona: ",personas)
        persona_txt = personas
        history_txt = self.preprocess_dial_history(dial_history + [user_input]) # PersonaChatBART put the user input together with dialogue history
        full_input_txt = f"{self.bos} [CONTEXT] {history_txt} [KNOWLEDGE] {persona_txt} {self.eos}"
        # print(full_input_txt)
        full_input_ids = self.tokenizer.encode(full_input_txt,return_tensors='pt', truncation=True, max_length=128).to(device)
        conversation_ids = self.model.generate(full_input_ids, max_length=1000, pad_token_id=self.eos_id)
        bot_respond = self.tokenizer.batch_decode(conversation_ids, skip_special_tokens=True)[0]
        return bot_respond


class Model2:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Model_name", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("Model_name")
        self.model.to(device)
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
    
    def preprocess_persona(self,personas):
        return " ".join(personas)
    
    def preprocess_dial_history(self,dial_history):
        return " ".join(dial_history)
    
    def response(self,user_input,dial_history,personas):
        # print("Persona: ", personas)
        persona_txt = personas
        history_txt = self.preprocess_dial_history(dial_history)
        full_input_txt = f"{persona_txt}{self.eos}{history_txt}{self.eos}{user_input}{self.eos}"
        # print("Full Input txt: ",full_input_txt)
        full_input_ids = self.tokenizer.encode(full_input_txt,return_tensors='pt', truncation=True, max_length=128).to(device)
        conversation_ids = self.model.generate(full_input_ids, max_length=1000, pad_token_id=self.eos_id)
        bot_respond = self.tokenizer.decode(conversation_ids[:, full_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return bot_respond


class QcModel:
    def __init__(self):
        '''
        Please put "degraded_random_responses_filtered.txt" and "utils.py" in the same directory.
        You can also set `qc_response_file` as the absolute path to "degraded_random_responses_filtered.txt" file.
        '''
        qc_response_file = path(__file__).parent.joinpath("degraded_random_responses_filtered.txt")
        with open(qc_response_file) as f:
            self.all_qc_responses = f.read().strip().splitlines()
        
    def response(self, *args, **kwarg):
        random.seed() # reset seed 
        return random.choice(self.all_qc_responses)


class PersonaChatBART:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bart-base-en-persona-chat")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("DeepPavlov/bart-base-en-persona-chat")
        self.model.to(device)
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
        self.bos = self.tokenizer.bos_token
    
    def preprocess_persona(self,personas):
        return " ".join(personas)
  
    def preprocess_dial_history(self,dial_history):
        return " EOS ".join(dial_history)
        
    def response(self,user_input,dial_history,personas):
        persona_txt = personas
        history_txt = self.preprocess_dial_history(dial_history + [user_input]) # PersonaChatBART put the user input together with dialogue history
        full_input_txt = f"{self.bos} [CONTEXT] {history_txt} [KNOWLEDGE] {persona_txt} {self.eos}"

        full_input_ids = self.tokenizer.encode(full_input_txt,return_tensors='pt', truncation=True, max_length=128).to(device)
        conversation_ids = self.model.generate(full_input_ids, max_length=1000, pad_token_id=self.eos_id)
        bot_respond = self.tokenizer.batch_decode(conversation_ids, skip_special_tokens=True)[0]
        return bot_respond

class PersonaChatGPT:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT")
        self.model = GPT2LMHeadModel.from_pretrained("af1tang/personaGPT")
        self.model.to(device)
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
        
    def preprocess_persona(self,personas):
        # personas_fact = [f'Fact: {p}' for p in personas]
        personas_fact = [f'{p}{self.eos}' for p in personas]
        full_persona = ''.join(['<|p2|>'] + personas_fact + ['<|sep|>'] + ['<|start|>'])
        return full_persona
        
    def preprocess_dial_history(self,dial_history):
        return self.eos.join(dial_history)
    
    def response(self,user_input,dial_history,personas):
        persona_txt = personas
        history_txt = self.preprocess_dial_history(dial_history)
        full_input_txt = f"{persona_txt}{history_txt}{self.eos}{user_input}{self.eos}"
        full_input_ids = self.tokenizer.encode(full_input_txt,return_tensors='pt', truncation=True, max_length=128).to(device)
        # print(full_input_ids)
        conversation_ids = self.model.generate(full_input_ids, max_length=1000, pad_token_id=self.eos_id)
        # print(conversation_ids)
        bot_respond = self.tokenizer.decode(conversation_ids[:, full_input_ids.shape[-1]:][0], skip_special_tokens=True)
        # print(bot_respond)
        return bot_respond

class VanillaDialoGPT:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
    
    def preprocess_persona(self,personas):
        return " ".join(personas)
    
    def preprocess_dial_history(self,dial_history):
        return " ".join(dial_history)
    
    def response(self,user_input,dial_history,personas):
        persona_txt = personas
        history_txt = self.preprocess_dial_history(dial_history)
        full_input_txt = f"{persona_txt}{self.eos}{history_txt}{self.eos}{user_input}{self.eos}"
        full_input_ids = self.tokenizer.encode(full_input_txt,return_tensors='pt',truncation=True, max_length=128)
        conversation_ids = self.model.generate(full_input_ids, max_length=1000, pad_token_id=self.eos_id)
        bot_respond = self.tokenizer.decode(conversation_ids[:, full_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return bot_respond

class VanillaBlenderbotSmall:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
        self.model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
        self.bos = self.tokenizer.bos_token
    
    def preprocess_persona(self,personas):
        return "</s> <s>".join(personas)
  
    def preprocess_dial_history(self,dial_history):
        return "</s> <s>".join(dial_history)
    
    def response(self,user_input,dial_history,personas):
        persona_txt = personas
        history_txt = self.preprocess_dial_history(dial_history)
        full_input_txt = f"{self.bos}{persona_txt}{self.eos}{history_txt}{self.eos}{user_input}{self.eos}"
        full_input_ids = self.tokenizer([full_input_txt],return_tensors='pt',truncation=True, max_length=128).to(device)
        conversation_ids = self.model.generate(**full_input_ids, max_length=1000, pad_token_id=self.eos_id)
        bot_respond = self.tokenizer.batch_decode(conversation_ids, skip_special_tokens=True)[0]
        return bot_respond



class Blenderbot:
    def __init__(self):
        self.tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        self.model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
        self.bos = self.tokenizer.bos_token
    
    def preprocess_persona(self,personas):
        return "</s> <s>".join(personas)
  
    def preprocess_dial_history(self,dial_history):
        return "</s> <s>".join(dial_history)
    
    def response(self,user_input,dial_history,personas):
        persona_txt = personas
        history_txt = self.preprocess_dial_history(dial_history)
        full_input_txt = f"{self.bos}{persona_txt}{self.eos}{history_txt}{self.eos}{user_input}{self.eos}"
        full_input_ids = self.tokenizer([full_input_txt],return_tensors='pt', truncation=True, max_length=128)
        conversation_ids = self.model.generate(**full_input_ids, max_length=1000, pad_token_id=self.eos_id)
        bot_respond = self.tokenizer.batch_decode(conversation_ids, skip_special_tokens=True)[0]
        return bot_respond


class MGodel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
    
    def preprocess_persona(self,personas):
        return " ".join(personas)
    
    def preprocess_dial_history(self,dial_history):
        return " ".join(dial_history)
    
    def response(self,user_input,dial_history,personas):
        persona_txt = personas
        history_txt = self.preprocess_dial_history(dial_history)
        full_input_txt = f"{persona_txt}{self.eos}{history_txt}{self.eos}{user_input}{self.eos}"
        full_input_ids = self.tokenizer.encode(full_input_txt,return_tensors='pt', truncation=True, max_length=128).to(device)
        conversation_ids = self.model.generate(full_input_ids, max_length=1000, pad_token_id=self.eos_id)
        bot_respond = self.tokenizer.decode(conversation_ids[:, full_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return bot_respond

class DialogGPT:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.model.to(device)
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
    
    def preprocess_persona(self,personas):
        return " ".join(personas)
    
    def preprocess_dial_history(self,dial_history):
        return " ".join(dial_history)
    
    def response(self,user_input,dial_history,personas):
        persona_txt = personas
        history_txt = self.preprocess_dial_history(dial_history)
        full_input_txt = f"{persona_txt}{self.eos}{history_txt}{self.eos}{user_input}{self.eos}"
        full_input_ids = self.tokenizer.encode(full_input_txt,return_tensors='pt', truncation=True, max_length=128).to(device)
        conversation_ids = self.model.generate(full_input_ids, max_length=1000, pad_token_id=self.eos_id)
        bot_respond = self.tokenizer.decode(conversation_ids[:, full_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return bot_respond 
