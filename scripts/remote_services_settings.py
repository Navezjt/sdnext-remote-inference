import modules.shared
from modules.shared import OptionInfo, options_section
import gradio as gr

from enum import Enum
RemoteService = Enum('RemoteService', ['Local', 'SDNext', 'StableHorde', 'OmniInfer'])

modules.shared.options_templates.update(options_section(('sdnext_remote_inference', "Remote Inference"),{
    'remote_inference_service': OptionInfo(RemoteService.Local.name, "Remote inference service", gr.Dropdown, lambda: {"choices": [e.name for e in RemoteService]}),

    'sdnext_api_endpoint': OptionInfo('http://127.0.0.1:7860', 'SD.Next API endpoint'),
    'horde_api_endpoint': OptionInfo('https://stablehorde.net/api', 'StableHorde API endpoint'),
    'omniinfer_api_endpoint': OptionInfo('https://api.omniinfer.io', 'OmniInfer API endpoint')
}))
#modules.shared.opts.quicksettings_list = modules.shared.opts.quicksettings_list.insert(0, 'remote_inference_service')    

def get_current_api_service():
    return RemoteService[modules.shared.opts.remote_inference_service]                           