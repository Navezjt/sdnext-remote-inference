import gradio as gr

from extension.utils_remote import get_current_api_service, RemoteService, ModelType
from extension.remote_extra_networks import get_models

class UIBinding:
    COMPONENTS_IDS = ['setting_remote_inference_service', 'txt2img_sampling', 'img2img_sampling', 'extras_upscaler_1', 'txt2img_generate']

    def __init__(self):
        self.initialized = False
        self.components = {key: None for key in UIBinding.COMPONENTS_IDS}
        self.components_default = {}

    def __getattribute__(self, attr):
        if attr in UIBinding.COMPONENTS_IDS:
            return self.components[attr]
        return super().__getattribute__(attr)
    
    def add_component(self, component):
        if component.elem_id in self.components:
            self.components[component.elem_id] = component

            config = {attr: getattr(component, attr) for attr in ['value', 'choices'] if hasattr(component, attr)}
            self.components_default[component.elem_id] = config
    
    def ready_for_update(self):
        if not self.initialized and all(self.components.values()):
            self.initialized = True
            return True
        return False
    
    def back_to_default(self, components):
        return tuple(gr.update(**self.components_default[component.elem_id]) for component in components)
    
uibindings = UIBinding()

def bind_component(component):
    uibindings.add_component(component)

    if uibindings.ready_for_update():
        uibindings.setting_remote_inference_service.change(fn=change_model_dropdowns, inputs=[uibindings.setting_remote_inference_service], outputs=[uibindings.txt2img_sampling, uibindings.img2img_sampling, uibindings.extras_upscaler_1])

        #txt2img_interface, etc.: estimation
 
def change_model_dropdowns(setting_remote_inference_service_value):
    service = RemoteService[setting_remote_inference_service_value]

    if service == RemoteService.StableHorde:
        sampler = gr.Dropdown.update(choices=get_models(ModelType.SAMPLER, service))
        upscaler = gr.Dropdown.update(choices=get_models(ModelType.UPSCALER, service))
        return (sampler, sampler, upscaler)
    
    return uibindings.back_to_default([uibindings.txt2img_sampling, uibindings.img2img_sampling, uibindings.extras_upscaler_1])
    
    
