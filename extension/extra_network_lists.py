import requests
import json
import html
import time

import modules
import lora

from extension.utils_remote import ModelType, RemoteService, get_current_api_service, get_default_endpoint, safeget

class ModelListFetchError(Exception):
    def __init__(self, model_type, service, error):
        super().__init__(f'Unable to fetch remote {model_type} list for {service}: {error}')

def log_info_model_count(model_type, api_service, count):
    modules.shared.log.info(f'Available {model_type.name.lower()}s: {api_service} items={count}')

cache = {}
def get_or_error(model_type, service, path):
    global cache
    cache_key = (service, path)
    if cache_key in cache:
        result, timestamp = cache[cache_key]
        if time.time() - timestamp <= modules.shared.opts.remote_model_browser_cache_time:
            return result

    endpoint = get_default_endpoint(service)
    try:
        response = requests.get(endpoint+path)
    except Exception as e:
        raise ModelListFetchError(model_type, service, e)
    if response.status_code != 200:
        raise ModelListFetchError(model_type, service, response.content)
    
    result = json.loads(response.content)
    cache[cache_key] = (result, time.time())
    return result

def get_remote(model_type: ModelType, service: RemoteService):
    try:
        #================================== SD.Next ==================================
        if service == RemoteService.SDNext:
            if model_type == ModelType.CHECKPOINT:
                model_list = get_or_error(model_type, service, "/sdapi/v1/sd-models")

                for model in sorted(model_list, key=lambda model: str.lower(model['model_name'])):
                    RemoteCheckpointInfo(model['model_name'], service)

            elif model_type == ModelType.LORA:
                lora_list = get_or_error(model_type, service, "/sdapi/v1/loras")

                for lora_model in sorted(lora_list, key=str.lower):
                    LoraOnRemote(lora_model)

            elif model_type == ModelType.TEXTUALINVERSION:
                pass

        #================================== Stable Horde ==================================
        elif service == RemoteService.StableHorde:
            if model_type == ModelType.CHECKPOINT:          
                model_list = get_or_error(model_type, service, "/v2/status/models")
                model_list = filter(lambda model: model['type'] == 'image', model_list)
                
                data = json.loads(requests.get('https://github.com/Haidra-Org/AI-Horde-image-model-reference/blob/main/stable_diffusion.json').content)
                data_models = json.loads(''.join(data['payload']['blob']['rawLines']))
                
                for model in sorted(model_list, key=lambda model: model['count'], reverse=True):
                    model_data = safeget(data_models, model['name'])
                    RemoteCheckpointInfo(f"{model['name']} ({model['count']})", service, safeget(model_data,'showcases',0), safeget(model_data,'description'))
        
            elif model_type == ModelType.LORA:
                pass
            elif model_type == ModelType.TEXTUALINVERSION:
                pass

        #================================== OmniInfer ==================================
        elif service == RemoteService.OmniInfer:
            if model_type == ModelType.CHECKPOINT:
                model_list = get_or_error(model_type, service, "/v2/models")
                model_list = model_list['data']['models']
                model_list = filter(lambda model: model['type'] == 'checkpoint', model_list)

                for model in sorted(model_list, key=lambda model: str.lower(model['name'])):
                    RemoteCheckpointInfo(model['name'], service, safeget(model, 'civitai_images', 0, 'url'))

            elif model_type == ModelType.LORA:          
                lora_list = get_or_error(model_type, service, "/v2/models")
                lora_list = lora_list['data']['models']
                lora_list = filter(lambda model: model['type'] == 'lora', lora_list)
                for lora_model in sorted(lora_list, key=lambda model: str.lower(model['name'])):
                    tags = {tag:0 for tag in lora_model['civitai_tags'].split(',')} if 'civitai_tags' in lora_model else {}
                    LoraOnRemote(lora_model['name'], safeget(lora_model, 'civitai_images', 0, 'url'), tags=tags)

            elif model_type == ModelType.TEXTUALINVERSION:
                pass
    except ModelListFetchError as e:
        modules.shared.log.error(e)

class PreviewDescriptionInfo():
    pass

def get_remote_preview_description_info(self, prev_desc_info):
    preview = prev_desc_info.preview_url or self.link_preview('html/card-no-preview.png')
    description = prev_desc_info.description or ''
    info = prev_desc_info.info or ''
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

def list_remote_models():
    api_service = get_current_api_service()

    modules.sd_models.checkpoints_list.clear()
    modules.sd_models.checkpoint_aliases.clear()

    get_remote(ModelType.CHECKPOINT, api_service)
    log_info_model_count(ModelType.CHECKPOINT, api_service, len(modules.sd_models.checkpoints_list))

def fake_reload_model_weights(sd_model=None, info=None, reuse_dict=False, op='model'):  
    checkpoint_info = info or modules.sd_models.select_checkpoint(op=op)
    modules.shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
    return True

def extra_networks_checkpoints_list_items(self):
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

    get_remote(ModelType.LORA, api_service)
    log_info_model_count(ModelType.LORA, api_service, len(lora.available_loras))

def extra_networks_loras_list_items(self):
    for name, lora_on_remote in lora.available_loras.items():
        preview, description, info = get_remote_preview_description_info(self, lora_on_remote)

        prompt = f" <lora:{lora_on_remote.get_alias()}:{modules.shared.opts.extra_networks_default_multiplier}>"
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