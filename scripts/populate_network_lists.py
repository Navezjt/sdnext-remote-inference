import requests
import json
import html

import modules
import modules.sd_models
import modules.ui_extra_networks_checkpoints
from modules import shared

from scripts.remote_services_settings import RemoteService, get_current_api_service

class RemoteCheckpointInfo(modules.sd_models.CheckpointInfo):
    def __init__(self, name, remote_service, preview_url=None, description=None, info=None):
        self.name = self.name_for_extra = self.model_name = self.title = name
        self.type = f"remote ({remote_service})"
        self.remote_service = remote_service

        self.preview_url = preview_url
        self.description = description
        self.info = info

        self.model_info = None
        self.metadata = {}

        self.sha256 = self.hash = self.shorthash = None
        self.filename = self.path = ''

        self.ids = [self.name]
        self.register()

class LoadModelListError(Exception):
    def __init__(self, error, remote_service):
        super(f'Unable to fetch remote model list for {remote_service}: {error}')

last_loaded_list = None

def safeget(dct, *keys):
    for key in keys:
        try:
            dct = dct[key]
        except (KeyError,TypeError,IndexError):
            return None
    return dct

def list_remote_models():
    api_service = get_current_api_service()

    modules.sd_models.checkpoints_list.clear()
    modules.sd_models.checkpoint_aliases.clear()

    if api_service == RemoteService.SDNext:
        model_list = requests.get(shared.opts.sdnext_api_endpoint+"/sdapi/v1/sd-models")
        if model_list.status_code != 200:
            raise LoadModelListError(api_service, model_list.content)
        
        model_list = json.loads(model_list.content)
        for model in sorted(model_list, key=lambda model: str.lower(model['model_name'])):
            RemoteCheckpointInfo(model['model_name'], RemoteService.SDNext)

    elif api_service == RemoteService.StableHorde:
        model_list = requests.get(shared.opts.horde_api_endpoint+"/v2/status/models", headers={'Client-Agent':'SD.Next Remote Inference:rolling:BinaryQuantumSoul'})
        if model_list.status_code != 200:
            raise LoadModelListError(api_service, model_list.content)
        
        data = json.loads(requests.get('https://github.com/Haidra-Org/AI-Horde-image-model-reference/blob/main/stable_diffusion.json').content)
        data_models = json.loads(''.join(data['payload']['blob']['rawLines']))
        
        model_list = json.loads(model_list.content)
        model_list = filter(lambda model: model['type'] == 'image', model_list)
        for model in reversed(sorted(model_list, key=lambda model: model['count'])):
            model_data = safeget(data_models, model['name'])
            RemoteCheckpointInfo(f"{model['name']} ({model['count']})", RemoteService.StableHorde, safeget(model_data,'showcases',0), safeget(model_data,'description'))
    
    elif api_service == RemoteService.OmniInfer:
        model_list = requests.get(shared.opts.omniinfer_api_endpoint+"/v2/models")
        if model_list.status_code != 200:
            raise LoadModelListError(api_service, model_list.content)
        
        model_list = json.loads(model_list.content)['data']['models']
        model_list = filter(lambda model: model['type'] == 'checkpoint', model_list)
        for model in sorted(model_list, key=lambda model: str.lower(model['name'])):
            RemoteCheckpointInfo(model['name'], RemoteService.OmniInfer, safeget(model, 'civitai_images', 0, 'url'))

    shared.log.info(f'Available models: {api_service} items={len(modules.sd_models.checkpoints_list)}')

def fake_reload_model_weights(sd_model=None, info=None, reuse_dict=False, op='model'):
    api_service = get_current_api_service()

    global last_loaded_list
    if last_loaded_list is None or last_loaded_list != api_service:
        list_remote_models()
        last_loaded_list = api_service
    
    checkpoint_info = info or modules.sd_models.select_checkpoint(op=op)
    shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
    return True

def get_remote_preview_description_info(self, checkpoint_info):
    preview = checkpoint_info.preview_url or self.link_preview('html/card-no-preview.png')
    description = checkpoint_info.description or ''
    info = ''
    return preview, description, info

def extra_networks_checkpoints_list_items(self):
    self.refresh()

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

modules.sd_models.list_models = list_remote_models
modules.sd_models.reload_model_weights = fake_reload_model_weights
modules.ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints.list_items = extra_networks_checkpoints_list_items