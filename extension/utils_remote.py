from enum import Enum

import modules

ModelType = Enum('ModelType', ['CHECKPOINT','LORA','TEXTUALINVERSION','CONTROLNET','VAE','UPSCALER','LYCORIS','HYPERNET'])
RemoteService = Enum('RemoteService', ['Local', 'SDNext', 'StableHorde', 'OmniInfer'])
default_endpoints = {
    RemoteService.SDNext: 'http://127.0.0.1:7860',
    RemoteService.StableHorde: 'https://stablehorde.net/api',
    RemoteService.OmniInfer: 'https://api.omniinfer.io'
}
setting_names = {
    RemoteService.SDNext: 'sdnext_api_endpoint',
    RemoteService.StableHorde: 'horde_api_endpoint',
    RemoteService.OmniInfer: 'omniinfer_api_endpoint'
}

def get_default_endpoint(remote_service):
    return modules.shared.opts.data.get(setting_names[remote_service], default_endpoints[remote_service])

def get_current_api_service():
    return RemoteService[modules.shared.opts.remote_inference_service]  

def safeget(dct, *keys):
    for key in keys:
        try:
            dct = dct[key]
        except (KeyError,TypeError,IndexError):
            return None
    return dct

import copy
def make_conditional_hook(func, replace_func):
    func_copy = copy.deepcopy(func)
    def wrap(*args, **kwargs):
        if get_current_api_service() == RemoteService.Local:
            return func_copy(*args, **kwargs)
        else:
            return replace_func(*args, **kwargs)
    return wrap
