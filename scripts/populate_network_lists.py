import requests
import json
import html

import modules.sd_models
import modules.ui_extra_networks_checkpoints
import lora
import ui_extra_networks_lora
from modules import shared

from extension.utils import RemoteService, get_default_endpoint, get_current_api_service, safeget, make_conditional_hook

class PreviewDescriptionInfo():
    pass

def get_remote_preview_description_info(self, prev_desc_info):
    preview = prev_desc_info.preview_url or self.link_preview('html/card-no-preview.png')
    description = prev_desc_info.description or ''
    info = ''
    return preview, description, info

#============================================= CHECKPOINTS =============================================
class RemoteCheckpointInfo(modules.sd_models.CheckpointInfo, PreviewDescriptionInfo):
    def __init__(self, name, remote_service, preview_url=None, description=None, info=None):
        self.name = self.name_for_extra = self.model_name = self.title = name
        self.type = f"remote ({remote_service})"
        self.ids = [self.name]

        self.preview_url = preview_url
        self.description = description
        self.info = info

        self.model_info = None
        self.metadata = {}
        self.sha256 = self.hash = self.shorthash = None
        self.filename = self.path = ''

        self.register()

class LoadModelListError(Exception):
    def __init__(self, error, remote_service):
        super(f'Unable to fetch remote model list for {remote_service}: {error}')

def list_remote_models():
    api_service = get_current_api_service()

    modules.sd_models.checkpoints_list.clear()
    modules.sd_models.checkpoint_aliases.clear()

    if api_service == RemoteService.SDNext:
        endpoint = get_default_endpoint('shared.opts.sdnext_api_endpoint', api_service)
        model_list = requests.get(endpoint+"/sdapi/v1/sd-models")
        if model_list.status_code != 200:
            raise LoadModelListError(api_service, model_list.content)
        
        model_list = json.loads(model_list.content)
        for model in sorted(model_list, key=lambda model: str.lower(model['model_name'])):
            RemoteCheckpointInfo(model['model_name'], api_service)

    elif api_service == RemoteService.StableHorde:
        endpoint = get_default_endpoint('shared.opts.horde_api_endpoint', api_service)
        model_list = requests.get(endpoint+"/v2/status/models", headers={'Client-Agent':'SD.Next Remote Inference:rolling:BinaryQuantumSoul'})
        if model_list.status_code != 200:
            raise LoadModelListError(api_service, model_list.content)
        
        data = json.loads(requests.get('https://github.com/Haidra-Org/AI-Horde-image-model-reference/blob/main/stable_diffusion.json').content)
        data_models = json.loads(''.join(data['payload']['blob']['rawLines']))
        
        model_list = json.loads(model_list.content)
        model_list = filter(lambda model: model['type'] == 'image', model_list)
        for model in reversed(sorted(model_list, key=lambda model: model['count'])):
            model_data = safeget(data_models, model['name'])
            RemoteCheckpointInfo(f"{model['name']} ({model['count']})", api_service, safeget(model_data,'showcases',0), safeget(model_data,'description'))
    
    elif api_service == RemoteService.OmniInfer:
        endpoint = get_default_endpoint('shared.opts.omniinfer_api_endpoint', api_service)
        model_list = requests.get(endpoint+"/v2/models")
        if model_list.status_code != 200:
            raise LoadModelListError(api_service, model_list.content)
        
        model_list = json.loads(model_list.content)['data']['models']
        model_list = filter(lambda model: model['type'] == 'checkpoint', model_list)
        for model in sorted(model_list, key=lambda model: str.lower(model['name'])):
            RemoteCheckpointInfo(model['name'], api_service, safeget(model, 'civitai_images', 0, 'url'))

    shared.log.info(f'Available models: {api_service} items={len(modules.sd_models.checkpoints_list)}')

last_loaded_list = None
def check_list_or_reload():
    global last_loaded_list
    api_service = get_current_api_service()
    if last_loaded_list  is None or last_loaded_list != api_service:
        list_remote_models()
        last_loaded_list = api_service

def fake_reload_model_weights(sd_model=None, info=None, reuse_dict=False, op='model'):
    check_list_or_reload()
    
    checkpoint_info = info or modules.sd_models.select_checkpoint(op=op)
    shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
    return True

def extra_networks_checkpoints_list_items(self):
    check_list_or_reload()

    checkpoint: RemoteCheckpointInfo
    for name, checkpoint in modules.sd_models.checkpoints_list.items():
        preview, description, info = get_remote_preview_description_info(self, checkpoint)

        yield {
            "name": checkpoint.name,
            "filename": checkpoint.name,
            "fullname": checkpoint.name,
            "hash": None,
            "preview": preview,
            "description": description,
            "info": info,
            "search_term": f'{checkpoint.name} /{checkpoint.type}/',
            "onclick": '"' + html.escape(f"""return selectCheckpoint({json.dumps(name)})""") + '"',
            "local_preview": None,
            "metadata": checkpoint.metadata,
        }

modules.sd_models.list_models = make_conditional_hook(modules.sd_models.list_models, list_remote_models)
modules.sd_models.reload_model_weights = make_conditional_hook(modules.sd_models.reload_model_weights, fake_reload_model_weights)
modules.ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints.list_items = make_conditional_hook(modules.ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints.list_items, extra_networks_checkpoints_list_items)

#============================================= LORAS =============================================       
class LoraOnRemote(lora.LoraOnDisk, PreviewDescriptionInfo):
    def __init__(self, name, preview_url=None, description=None, info=None, tags={}):
        self.name = self.alias = name
        self.filename = ''

        self.preview_url = preview_url
        self.description = description
        self.info = info

        self.tags = tags

        self.ssmd_cover_images = None
        self.metadata = {}
        self.hash = self.shorthash = None

        self.register()

    def register(self):
        lora.available_loras[self.name] = self
        if self.alias in lora.available_lora_aliases:
            lora.forbidden_lora_aliases[self.alias.lower()] = 1
        lora.available_lora_aliases[self.name] = self

def list_remote_loras():
    api_service = get_current_api_service()

    lora.available_loras.clear()
    lora.available_lora_aliases.clear()
    lora.forbidden_lora_aliases.clear()
    lora.available_lora_hash_lookup.clear()
    lora.forbidden_lora_aliases.update({"none": 1, "Addams": 1})

    if api_service == RemoteService.SDNext:
        endpoint = get_default_endpoint('shared.opts.sdnext_api_endpoint', api_service)
        lora_list = requests.get(endpoint+"/sdapi/v1/loras")
        if lora_list.status_code != 200:
            raise LoadModelListError(api_service, lora_list.content)
        
        lora_list = json.loads(lora_list.content)
        for lora_model in sorted(lora_list, key=str.lower):
            LoraOnRemote(lora_model)

    elif api_service == RemoteService.StableHorde:
        pass

    elif api_service == RemoteService.OmniInfer:
        endpoint = get_default_endpoint('shared.opts.omniinfer_api_endpoint', api_service)
        lora_list = requests.get(endpoint+"/v2/models")
        if lora_list.status_code != 200:
            raise LoadModelListError(api_service, lora_list.content)
        
        lora_list = json.loads(lora_list.content)['data']['models']
        lora_list = filter(lambda model: model['type'] == 'lora', lora_list)
        for lora_model in sorted(lora_list, key=lambda model: str.lower(model['name'])):
            tags = {tag:0 for tag in lora_model['civitai_tags'].split(',')} if 'civitai_tags' in lora_model else {}
            LoraOnRemote(lora_model['name'], safeget(lora_model, 'civitai_images', 0, 'url'), tags=tags)

def extra_networks_loras_list_items(self):
    for name, lora_on_remote in lora.available_loras.items():
        preview, description, info = get_remote_preview_description_info(self, lora_on_remote)

        prompt = f" <lora:{lora_on_remote.get_alias()}:{shared.opts.extra_networks_default_multiplier}>"
        prompt = json.dumps(prompt)

        yield {
            "name": name,
            "filename": name,
            "fullname": name,
            "hash": None,
            "preview": preview,
            "description": description,
            "info": info,
            "search_term": name,
            "prompt": prompt,
            "local_preview": None,
            "metadata": lora_on_remote.metadata,
            "tags": lora_on_remote.tags,
        }

lora.list_available_loras = make_conditional_hook(lora.list_available_loras, list_remote_loras)
ui_extra_networks_lora.ExtraNetworksPageLora.list_items = make_conditional_hook(ui_extra_networks_lora.ExtraNetworksPageLora.list_items, extra_networks_loras_list_items)