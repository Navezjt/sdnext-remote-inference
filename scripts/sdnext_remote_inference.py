import gradio as gr

import modules.sd_models
import modules.ui_extra_networks_checkpoints
import modules.textual_inversion.textual_inversion
import modules.ui_extra_networks_textual_inversion
import modules.processing
import modules.scripts_postprocessing
import modules.scripts
import modules.script_callbacks
import modules.shared
from modules.shared import OptionInfo, options_section 

from extension.utils_remote import make_conditional_hook, RemoteService, default_endpoints, endpoint_setting_names, apikey_setting_names, import_script_data
import extension.remote_extra_networks
import extension.remote_process
import extension.remote_balance
import extension.remote_postprocess
import extension.ui_populate

import ui_extra_networks_lora
import networks

def on_app_started(blocks, _app):
    # SCRIPT IMPORTS
    import_script_data({
        'controlnet': 'extensions-builtin/sd-webui-controlnet/scripts/controlnet.py',
        'rembg': 'extensions-builtin/stable-diffusion-webui-rembg/scripts/postprocessing_rembg.py',
        'codeformer': 'scripts/postprocessing_codeformer.py',
        'gfpgan': 'scripts/postprocessing_gfpgan.py',
        'upscale': 'scripts/postprocessing_upscale.py'
    })

    # EXTRA NETWORKS
    modules.sd_models.list_models = make_conditional_hook(modules.sd_models.list_models, extension.remote_extra_networks.list_remote_models)
    modules.ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints.list_items = make_conditional_hook(modules.ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints.list_items, extension.remote_extra_networks.extra_networks_checkpoints_list_items)

    networks.list_available_networks = make_conditional_hook(networks.list_available_networks, extension.remote_extra_networks.list_remote_loras)
    ui_extra_networks_lora.ExtraNetworksPageLora.list_items = make_conditional_hook(ui_extra_networks_lora.ExtraNetworksPageLora.list_items, extension.remote_extra_networks.extra_networks_loras_list_items)

    modules.textual_inversion.textual_inversion.EmbeddingDatabase.load_textual_inversion_embeddings = make_conditional_hook(modules.textual_inversion.textual_inversion.EmbeddingDatabase.load_textual_inversion_embeddings, extension.remote_extra_networks.list_remote_embeddings)
    modules.ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion.refresh = make_conditional_hook(modules.ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion.refresh, extension.remote_extra_networks.extra_networks_textual_inversions_refresh)
    modules.ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion.list_items = make_conditional_hook(modules.ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion.list_items, extension.remote_extra_networks.extra_networks_textual_inversions_list_items)

    # GENERATION
    modules.sd_models.reload_model_weights = make_conditional_hook(modules.sd_models.reload_model_weights, extension.remote_process.fake_reload_model_weights)
    modules.processing.process_images = make_conditional_hook(modules.processing.process_images, extension.remote_process.process_images)
    modules.scripts_postprocessing.ScriptPostprocessingRunner.run = make_conditional_hook(modules.scripts_postprocessing.ScriptPostprocessingRunner.run, extension.remote_postprocess.remote_run) 

    # UI
    with blocks:
        balance = gr.HTML(value='', elem_id='remote_inference_balance')
        clicker = gr.Button(value='', visible=False, elem_id='remote_inference_balance_click')
        clicker.click(extension.remote_balance.remote_balance_gradio_update, inputs=[], outputs=[balance], show_progress='hidden')

modules.script_callbacks.on_app_started(on_app_started)
modules.script_callbacks.after_process_callback(lambda p: extension.remote_balance.refresh_balance())
modules.script_callbacks.on_after_component(lambda component, **kwargs: extension.ui_populate.bind_component(component))

# SETTINGS
def on_ui_settings():
    modules.shared.options_templates.update(options_section(('sdnext_remote_inference', "Remote Inference"),{
        'remote_sdnext_sep': OptionInfo("<h2>SD.Next API</h2>", "", gr.HTML),
        endpoint_setting_names[RemoteService.SDNext]: OptionInfo(default_endpoints[RemoteService.SDNext], 'SD.Next API endpoint'),

        'remote_horde_sep': OptionInfo("<h2>Stable Horde</h2>", "", gr.HTML),
        endpoint_setting_names[RemoteService.StableHorde]: OptionInfo(default_endpoints[RemoteService.StableHorde], 'StableHorde API endpoint'),
        apikey_setting_names[RemoteService.StableHorde]: OptionInfo('0000000000', 'StableHorde API Key (\'0000000000\' for anonymous)', gr.Textbox, {"type": "password"}),
        'horde_apikey_url': OptionInfo('<p>Get an API key <a href="https://stablehorde.net/register">here</a></p>', "", gr.HTML),
        'horde_nsfw': OptionInfo(False, "Enable NSFW generation (will skip anti-nsfw workers)"),
        'horde_censor_nsfw': OptionInfo(False, "Censor NSFW generations"),
        'horde_trusted_workers': OptionInfo(False, "Only trusted workers (slower but less risk)"),
        'horde_slow_workers': OptionInfo(True, "Allow slow workers (extra kudos cost if disabled)"),
        'horde_workers': OptionInfo('', "Comma-separated list of allowed/disallowed workers (max 5)"),
        'horde_worker_blacklist': OptionInfo(False, "Above list is a blacklist instead of a whitelist"),
        'horde_share_laion': OptionInfo(False, 'Share images with LAION for improving their dataset, reduce your kudos consumption by 2 (always True for anonymous users)'),

        'remote_omniinfer_sep': OptionInfo("<h2>OmniInfer</h2>", "", gr.HTML),
        endpoint_setting_names[RemoteService.OmniInfer]: OptionInfo(default_endpoints[RemoteService.OmniInfer], 'OmniInfer API endpoint'),
        apikey_setting_names[RemoteService.OmniInfer]: OptionInfo('', 'OmniInfer API Key', gr.Textbox, {"type": "password"}),
        'omniinfer_apikey_url': OptionInfo('<p>Get an API key <a href="https://www.omniinfer.io/dashboard/key">here</a></p>', "", gr.HTML),

        'remote_general_sep': OptionInfo("<h2>Other Settings</h2>", "", gr.HTML),
        'remote_inference_service': OptionInfo(RemoteService.Local.name, "Remote inference service", gr.Dropdown, {"choices": [e.name for e in RemoteService]}),
        'remote_balance_cache_time': OptionInfo(60, 'Cache time (in seconds) for remote balance api calls', gr.Slider, {"minimum": 60, "maximum": 3600, "step": 60}),
        'remote_extra_networks_cache_time': OptionInfo(600, 'Cache time (in seconds) for remote extra networks api calls', gr.Slider, {"minimum": 60, "maximum": 3600, "step": 60}),
        'show_remote_balance': OptionInfo(True, "Show top right available balance box"),
        'show_nsfw_models': OptionInfo(False, "Show NSFW networks (StableHorde/OmniInfer)")
    }))

    if modules.shared.opts.quicksettings_list[0] != 'remote_inference_service':
        modules.shared.opts.quicksettings_list.insert(0, 'remote_inference_service') 

modules.script_callbacks.on_ui_settings(on_ui_settings)