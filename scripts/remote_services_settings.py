import modules.shared
from modules.shared import OptionInfo, options_section
from modules import script_callbacks
import gradio as gr       

from extension.utils import RemoteService, default_endpoints

def on_ui_settings():
    modules.shared.options_templates.update(options_section(('sdnext_remote_inference', "Remote Inference"),{
    'remote_inference_service': OptionInfo(RemoteService.Local.name, "Remote inference service", gr.Dropdown, lambda: {"choices": [e.name for e in RemoteService]}),

    'sdnext_api_endpoint': OptionInfo(default_endpoints[RemoteService.SDNext], 'SD.Next API endpoint'),
    'horde_api_endpoint': OptionInfo(default_endpoints[RemoteService.StableHorde], 'StableHorde API endpoint'),
    'omniinfer_api_endpoint': OptionInfo(default_endpoints[RemoteService.OmniInfer], 'OmniInfer API endpoint')
    }))

    if modules.shared.opts.quicksettings_list[0] != 'remote_inference_service':
        modules.shared.opts.quicksettings_list.insert(0, 'remote_inference_service') 

script_callbacks.on_ui_settings(on_ui_settings)