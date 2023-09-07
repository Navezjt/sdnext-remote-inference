import requests
import json
import html

import modules
import modules.sd_models
import modules.ui_extra_networks_checkpoints
from modules import shared

#=================== tmp here ===================
from enum import Enum
RemoteService = Enum('RemoteService', ['LOCAL', 'SDNEXT', 'HORDE', 'OMNIINFER'])

sdnext_api_endpoint = 'https://127.0.0.1:7860'
horde_api_endpoint = "https://stablehorde.net/api"
omniinfer_api_endpoint = 'https://api.omniinfer.io'

horde_client_agent = 'SD.Next Remote Inference:rolling:BinaryQuantumSoul'
#=================== tmp here ===================

api_service = RemoteService.HORDE

class RemoteCheckpointInfo(modules.sd_models.CheckpointInfo):
    def __init__(self, name, remote_service, civitai_id=None):
        self.name = self.name_for_extra = self.model_name = self.title = name
        self.type = f"remote ({remote_service})"

        self.remote_service = remote_service
        self.civitai_id = civitai_id

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

def list_remote_models():
    modules.sd_models.checkpoints_list.clear()
    modules.sd_models.checkpoint_aliases.clear()

    if api_service == RemoteService.SDNEXT:
        model_list = requests.get(api_providers.sdnext_api_endpoint+"/sdapi/v1/sd-models")
        if model_list.status_code != 200:
            raise LoadModelListError(api_service, model_list.content)
        
        model_list = json.loads(model_list.content)
        for model in sorted(model_list, key=lambda model: str.lower(model['model_name'])):
            RemoteCheckpointInfo(model['model_name'], RemoteService.SDNEXT)

    elif api_service == RemoteService.HORDE:
        model_list = requests.get(api_providers.horde_api_endpoint+"/v2/status/models", headers={'Client-Agent':api_providers.horde_client_agent})
        if model_list.status_code != 200:
            raise LoadModelListError(api_service, model_list.content)
        
        model_list = json.loads(model_list.content)
        model_list = filter(lambda model: model['type'] == 'image', model_list)
        for model in sorted(model_list, key=lambda model: str.lower(model['name'])):
            RemoteCheckpointInfo(model['name'], RemoteService.HORDE)
    
    elif api_service == RemoteService.OMNIINFER:
        model_list = requests.get(api_providers.omniinfer_api_endpoint+"/v2/models")
        if model_list.status_code != 200:
            raise LoadModelListError(api_service, model_list.content)
        
        model_list = json.loads(model_list.content)['data']['models']
        model_list = filter(lambda model: model['type'] == 'checkpoint', model_list)
        for model in sorted(model_list, key=lambda model: str.lower(model['name'])):
            RemoteCheckpointInfo(model['name'], RemoteService.OMNIINFER, getattr(model, 'civitai_model_id', None))

    shared.log.info(f'Available models: {api_service} items={len(modules.sd_models.checkpoints_list)}')

def fake_reload_model_weights(sd_model=None, info=None, reuse_dict=False, op='model'):
    global last_loaded_list
    if last_loaded_list is None or last_loaded_list != api_service:
        list_remote_models()
        last_loaded_list = api_service
    
    checkpoint_info = info or modules.sd_models.select_checkpoint(op=op)
    shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
    return True

def get_remote_preview_description_info(self, checkpoint_info):
    if checkpoint_info.remote_service == RemoteService.HORDE:
        pass

    return self.link_preview('html/logo-bg-1.jpg'), '', ''

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

if api_service != RemoteService.LOCAL:
    modules.sd_models.list_models = list_remote_models
    modules.sd_models.reload_model_weights = fake_reload_model_weights
    modules.ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints.list_items = extra_networks_checkpoints_list_items