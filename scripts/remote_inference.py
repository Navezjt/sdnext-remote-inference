import modules.sd_models
import modules.ui_extra_networks_checkpoints
import lora
import ui_extra_networks_lora
import modules.textual_inversion.textual_inversion
import modules.ui_extra_networks_textual_inversion

import modules.processing

from extension.utils_remote import make_conditional_hook
import extension.remote_extra_networks
import extension.remote_process

import modules.script_callbacks
import modules.shared
from modules.shared import OptionInfo, options_section
import gradio as gr       

from extension.utils_remote import RemoteService, default_endpoints, endpoint_setting_names, apikey_setting_names

def on_app_started(*args, **kwargs):
    # EXTRA NETWORKS
    modules.sd_models.list_models = make_conditional_hook(modules.sd_models.list_models, extension.remote_extra_networks.list_remote_models)
    modules.ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints.list_items = make_conditional_hook(modules.ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints.list_items, extension.remote_extra_networks.extra_networks_checkpoints_list_items)

    lora.list_available_loras = make_conditional_hook(lora.list_available_loras, extension.remote_extra_networks.list_remote_loras)
    ui_extra_networks_lora.ExtraNetworksPageLora.list_items = make_conditional_hook(ui_extra_networks_lora.ExtraNetworksPageLora.list_items, extension.remote_extra_networks.extra_networks_loras_list_items)

    modules.textual_inversion.textual_inversion.EmbeddingDatabase.load_textual_inversion_embeddings = make_conditional_hook(modules.textual_inversion.textual_inversion.EmbeddingDatabase.load_textual_inversion_embeddings, extension.remote_extra_networks.list_remote_embeddings)
    modules.ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion.refresh = make_conditional_hook(modules.ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion.refresh, extension.remote_extra_networks.extra_networks_textual_inversions_refresh)
    modules.ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion.list_items = make_conditional_hook(modules.ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion.list_items, extension.remote_extra_networks.extra_networks_textual_inversions_list_items)

    # GENERATION
    modules.sd_models.reload_model_weights = make_conditional_hook(modules.sd_models.reload_model_weights, extension.remote_process.fake_reload_model_weights)
    modules.processing.process_images = make_conditional_hook(modules.processing.process_images, extension.remote_process.process_images)

modules.script_callbacks.on_app_started(on_app_started)

# SETTINGS
def on_ui_settings():
    modules.shared.options_templates.update(options_section(('sdnext_remote_inference', "Remote Inference"),{
    'remote_inference_service': OptionInfo(RemoteService.Local.name, "Remote inference service", gr.Dropdown, {"choices": [e.name for e in RemoteService]}),

    endpoint_setting_names[RemoteService.SDNext]: OptionInfo(default_endpoints[RemoteService.SDNext], 'SD.Next API endpoint'),
    endpoint_setting_names[RemoteService.StableHorde]: OptionInfo(default_endpoints[RemoteService.StableHorde], 'StableHorde API endpoint'),
    apikey_setting_names[RemoteService.StableHorde]: OptionInfo('', 'StableHorde API Key', gr.Textbox, {"type": "password"}),
    endpoint_setting_names[RemoteService.OmniInfer]: OptionInfo(default_endpoints[RemoteService.OmniInfer], 'OmniInfer API endpoint'),
    apikey_setting_names[RemoteService.OmniInfer]: OptionInfo('', 'OmniInfer API Key', gr.Textbox, {"type": "password"}),

    'remote_model_browser_cache_time': OptionInfo(600, 'Cache time (in seconds) for remote extra networks api calls', gr.Slider, {"minimum": 1, "maximum": 3600, "step": 1}),
    'skip_nsfw_models': OptionInfo(True, "Don't show (most of) NSFW networks (StableHorde/OmniInfer)")
    }))

    if modules.shared.opts.quicksettings_list[0] != 'remote_inference_service':
        modules.shared.opts.quicksettings_list.insert(0, 'remote_inference_service') 

modules.script_callbacks.on_ui_settings(on_ui_settings)