import requests
import json
import html
import time

import modules.shared
import modules.sd_models
import modules.textual_inversion
import modules.sd_hijack
import modules.ui_extra_networks
log = modules.shared.log
import lora

from extension.utils_remote import ModelType, RemoteService, get_current_api_service, get_default_endpoint, safeget

class ModelListFetchError(Exception):
    def __init__(self, model_type, service, error):
        super().__init__(f'Unable to fetch remote {model_type} list for {service}: {error}')

def log_debug_model_list(model_type, api_service):
    log.debug(f'RI: Listing {model_type.name.lower()}s from {api_service}')

def log_info_model_count(model_type, api_service, count):
    log.info(f'RI: Available {model_type.name.lower()}s: {api_service} items={count}')

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
                print(model_list)

                for model in sorted(model_list, key=lambda model: str.lower(model['model_name'])):
                    RemoteCheckpointInfo(model['model_name'], service)

            elif model_type == ModelType.LORA:
                model_list = get_or_error(model_type, service, "/sdapi/v1/loras")

                for model in sorted(model_list, key=str.lower):
                    RemoteLora(model)

            elif model_type == ModelType.TEXTUALINVERSION:
                model_list = get_or_error(model_type, service, "/sdapi/v1/embeddings")

                for model in sorted(model_list, key=str.lower):
                    pass#RemoteEmbedding(model)

        #================================== Stable Horde ==================================
        elif service == RemoteService.StableHorde:
            if model_type == ModelType.CHECKPOINT:          
                model_list = get_or_error(model_type, service, "/v2/status/models")
                model_list = filter(lambda model: model['type'] == 'image', model_list)
                
                data = json.loads(requests.get('https://github.com/Haidra-Org/AI-Horde-image-model-reference/blob/main/stable_diffusion.json').content)
                data_models = json.loads(''.join(data['payload']['blob']['rawLines']))
                
                for model in sorted(model_list, key=lambda model: model['count'], reverse=True):
                    model_data = safeget(data_models, model['name'])
                    if not safeget(model_data, 'nsfw') or not modules.shared.opts.skip_nsfw_models: 
                        RemoteCheckpointInfo(f"{model['name']} ({model['count']})", service, safeget(model_data,'showcases',0), safeget(model_data,'description'))
        
            elif model_type == ModelType.LORA:
                pass
            elif model_type == ModelType.TEXTUALINVERSION:
                pass

        #================================== OmniInfer ==================================
        elif service == RemoteService.OmniInfer:
            if not model_type in [ModelType.CHECKPOINT, ModelType.LORA, ModelType.TEXTUALINVERSION]:
                return

            model_list = get_or_error(model_type, service, "/v2/models")
            model_list = model_list['data']['models']
            if modules.shared.opts.skip_nsfw_models:
                model_list = list(filter(lambda x: not x['civitai_nsfw'], model_list))
                for model in model_list:
                    if 'civitai_images' in model:
                        model['civitai_images'] = list(filter(lambda img: img['nsfw'] == 'None', model['civitai_images']))

            if model_type == ModelType.CHECKPOINT:
                model_list = filter(lambda model: model['type'] == 'checkpoint', model_list)
                for model in sorted(model_list, key=lambda model: str.lower(model['name'])):
                    RemoteCheckpointInfo(model['name'], service, safeget(model, 'civitai_images', 0, 'url'))

            elif model_type == ModelType.LORA:          
                model_list = filter(lambda model: model['type'] == 'lora', model_list)
                for model in sorted(model_list, key=lambda model: str.lower(model['name'])):
                    tags = {tag:0 for tag in model['civitai_tags'].split(',')} if 'civitai_tags' in model else {}
                    RemoteLora(model['name'], safeget(model, 'civitai_images', 0, 'url'), tags=tags)

            elif model_type == ModelType.TEXTUALINVERSION:
                model_list = filter(lambda model: model['type'] == 'textualinversion', model_list)
                for model in sorted(model_list, key=lambda model: str.lower(model['name'])):
                    RemoteEmbedding(model['name'], safeget(model, 'civitai_images', 0, 'url'))

    except ModelListFetchError as e:
        log.error(f'RI: {e}')

class PreviewDescriptionInfo():
    no_preview = modules.ui_extra_networks.ExtraNetworksPage.link_preview(None, 'html/card-no-preview.png')

    def __init__(self, preview_url=None, description=None, info=None):
        self.preview = preview_url or PreviewDescriptionInfo.no_preview
        self.description = description
        self.info = info

#============================================= CHECKPOINTS =============================================
class RemoteCheckpointInfo(modules.sd_models.CheckpointInfo, PreviewDescriptionInfo):
    def __init__(self, name, remote_service, preview_url=None, description=None, info=None):
        PreviewDescriptionInfo.__init__(self, preview_url, description, info)

        self.name = self.name_for_extra = self.model_name = self.title = name
        self.type = f"remote ({remote_service})"
        self.ids = [self.name]

        self.model_info = None
        self.metadata = {}
        self.sha256 = self.hash = self.shorthash = None
        self.filename = self.path = ''

        self.register()

def list_remote_models():
    api_service = get_current_api_service()

    modules.sd_models.checkpoints_list.clear()
    modules.sd_models.checkpoint_aliases.clear()

    log_debug_model_list(ModelType.CHECKPOINT, api_service)
    get_remote(ModelType.CHECKPOINT, api_service)
    log_info_model_count(ModelType.CHECKPOINT, api_service, len(modules.sd_models.checkpoints_list))

def fake_reload_model_weights(sd_model=None, info=None, reuse_dict=False, op='model'):
    try:
        checkpoint_info = info or modules.sd_models.select_checkpoint(op=op)
        modules.shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
        return True
    except StopIteration:
        return False

def extra_networks_checkpoints_list_items(self):
    for name, checkpoint in modules.sd_models.checkpoints_list.items():
        yield {
            "name": name,
            "filename": name,
            "fullname": name,
            "hash": None,
            "preview": checkpoint.preview,
            "description": checkpoint.description,
            "info": checkpoint.info,
            "search_term": f'{name} /{checkpoint.type}/',
            "onclick": '"' + html.escape(f"""return selectCheckpoint({json.dumps(name)})""") + '"',
            "local_preview": None,
            "metadata": checkpoint.metadata,
        }

#============================================= LORAS =============================================       
class RemoteLora(lora.LoraOnDisk, PreviewDescriptionInfo):
    def __init__(self, name, preview_url=None, description=None, info=None, tags={}):
        PreviewDescriptionInfo.__init__(self, preview_url, description, info)

        self.name = self.alias = name
        self.filename = ''

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

    log_debug_model_list(ModelType.LORA, api_service)
    get_remote(ModelType.LORA, api_service)
    log_info_model_count(ModelType.LORA, api_service, len(lora.available_loras))

def extra_networks_loras_list_items(self):
    for name, remote_lora in lora.available_loras.items():
        prompt = f" <lora:{remote_lora.get_alias()}:{modules.shared.opts.extra_networks_default_multiplier}>"
        prompt = json.dumps(prompt)

        yield {
            "name": name,
            "filename": name,
            "fullname": name,
            "hash": None,
            "preview": remote_lora.preview,
            "description": remote_lora.description,
            "info": remote_lora.info,
            "search_term": name,
            "prompt": prompt,
            "local_preview": None,
            "metadata": remote_lora.metadata,
            "tags": remote_lora.tags,
        }

#============================================= EMBEDDINGS =============================================
class RemoteEmbedding(modules.textual_inversion.textual_inversion.Embedding, PreviewDescriptionInfo):
    def __init__(self, name, preview_url=None, description=None, info=None):
        super().__init__(None, name)
        PreviewDescriptionInfo.__init__(self, preview_url, description, info)

        self.filename = ''

        self.register()

    def register(self):
        modules.sd_hijack.model_hijack.embedding_db.word_embeddings[self.name] = self

def extra_networks_textual_inversions_refresh(self):
    modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

def list_remote_embeddings(self, force_reload=False):
    api_service = get_current_api_service()

    self.ids_lookup.clear()
    self.word_embeddings.clear()
    self.skipped_embeddings.clear()
    self.embeddings_used.clear()
    self.expected_shape = None
    self.embedding_dirs.clear()

    log_debug_model_list(ModelType.TEXTUALINVERSION, api_service)
    get_remote(ModelType.TEXTUALINVERSION, api_service)
    log_info_model_count(ModelType.TEXTUALINVERSION, api_service, len(self.word_embeddings))

def extra_networks_textual_inversions_list_items(self):
    for name, embedding in modules.sd_hijack.model_hijack.embedding_db.word_embeddings.items():
        yield {
            "name": name,
            "filename": name,
            "preview": embedding.preview,
            "description": embedding.description,
            "info": embedding.info,
            "search_term": name,
            "prompt": name,
            "local_preview": None,
        }
