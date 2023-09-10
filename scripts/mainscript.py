import modules.sd_models
import modules.ui_extra_networks_checkpoints
import lora
import ui_extra_networks_lora

import extension.network_lists
from extension.remote_services_utils import make_conditional_hook

modules.sd_models.list_models = make_conditional_hook(modules.sd_models.list_models, extension.network_lists.list_remote_models)
modules.sd_models.reload_model_weights = make_conditional_hook(modules.sd_models.reload_model_weights, extension.network_lists.fake_reload_model_weights)
modules.ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints.list_items = make_conditional_hook(modules.ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints.list_items, extension.network_lists.extra_networks_checkpoints_list_items)

lora.list_available_loras = make_conditional_hook(lora.list_available_loras, extension.network_lists.list_remote_loras)
ui_extra_networks_lora.ExtraNetworksPageLora.list_items = make_conditional_hook(ui_extra_networks_lora.ExtraNetworksPageLora.list_items, extension.network_lists.extra_networks_loras_list_items)

import modules
from modules.shared import OptionInfo, options_section
import gradio as gr       

from extension.remote_services_utils import RemoteService, default_endpoints

def on_ui_settings():
    modules.shared.options_templates.update(options_section(('sdnext_remote_inference', "Remote Inference"),{
    'remote_inference_service': OptionInfo(RemoteService.Local.name, "Remote inference service", gr.Dropdown, lambda: {"choices": [e.name for e in RemoteService]}),

    'sdnext_api_endpoint': OptionInfo(default_endpoints[RemoteService.SDNext], 'SD.Next API endpoint'),
    'horde_api_endpoint': OptionInfo(default_endpoints[RemoteService.StableHorde], 'StableHorde API endpoint'),
    'omniinfer_api_endpoint': OptionInfo(default_endpoints[RemoteService.OmniInfer], 'OmniInfer API endpoint')
    }))

    if modules.shared.opts.quicksettings_list[0] != 'remote_inference_service':
        modules.shared.opts.quicksettings_list.insert(0, 'remote_inference_service') 

modules.script_callbacks.on_ui_settings(on_ui_settings)