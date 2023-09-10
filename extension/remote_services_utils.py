from enum import Enum

from modules import shared

RemoteService = Enum('RemoteService', ['Local', 'SDNext', 'StableHorde', 'OmniInfer'])
default_endpoints = {
    RemoteService.SDNext: 'http://127.0.0.1:7860',
    RemoteService.StableHorde: 'https://stablehorde.net/api',
    RemoteService.OmniInfer: 'https://api.omniinfer.io'
}

def get_default_endpoint(opt, remote_service):
    return shared.opts.data.get(opt, default_endpoints[remote_service])

def get_current_api_service():
    return RemoteService[shared.opts.remote_inference_service]  

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
        api_service = get_current_api_service()
        if api_service == RemoteService.Local:
            return func_copy(*args, **kwargs)
        else:
            return replace_func(*args, **kwargs)
    return wrap
