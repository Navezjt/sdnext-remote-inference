import modules.processing
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, Processed
import modules.shared
from modules.shared import state, log

from extension.utils_remote import encode_image, decode_image, download_images, get_current_api_service, get_image, request_or_error, RemoteService, get_api_key, stable_horde_samplers, stable_horde_client

import json
import time

class RemoteModel:
    def __init__(self, checkpoint_info):
        self.sd_checkpoint_info = checkpoint_info
        self.sd_model_hash = ''
        self.is_sdxl = False

def fake_reload_model_weights(sd_model=None, info=None, reuse_dict=False, op='model'):
    try:
        checkpoint_info = info or modules.sd_models.select_checkpoint(op=op)
        modules.shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
        return RemoteModel(checkpoint_info)
    except StopIteration:
        return None

class GenerateRemoteError(Exception):
    def __init__(self, service, error):
        super().__init__(f'RI: Unable to remotely infer task for {service}: {error}')

def generate_images(service: RemoteService, p: StableDiffusionProcessing) -> Processed:
    p.seed = int(modules.processing.get_fixed_seed(p.seed))
    p.subseed = int(modules.processing.get_fixed_seed(p.seed))
    p.prompt = modules.shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = modules.shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)

    #================================== SD.Next ==================================
    if service == RemoteService.SDNext:
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            pass
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            pass

    #================================== Stable Horde ==================================
    # Copyright NatanJunges
    elif service ==  RemoteService.StableHorde:
        n = p.n_iter*p.batch_size

        headers = {
            "apikey": get_api_key(service),
            "Client-Agent": stable_horde_client,
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": f"{p.prompt}###{p.negative_prompt}" if len(p.negative_prompt) > 0 else p.prompt,
            "params": {
                "sampler_name": stable_horde_samplers.get(p.sampler_name, "k_euler_a"),
                "cfg_scale": p.cfg_scale,
                "clip_skip": p.clip_skip,
                "seed": str(p.seed),
                "seed_variation": max(1, min(p.subseed, 1000)),
                "height": p.height,
                "width": p.width,
                "steps": p.steps,
                "n": n,
                "tiling": p.tiling
            },
            "models": [modules.shared.sd_model.sd_checkpoint_info.filename]
        }

        if " Karras" in p.sampler_name:
            payload["params"]["karras"] = True

        if isinstance(p, StableDiffusionProcessingTxt2Img):
            if p.enable_hr:
                payload["params"]["hires_fix"] = True
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            payload["source_image"] = encode_image(p.init_images[0])
            payload["params"]["denoising_strength"] = p.denoising_strength

            if p.image_mask:
                payload["source_processing"] = "inpainting"
                payload["source_mask"] = encode_image(p.image_mask)
        else:
            return

        # todo
        # if nsfw:
        #     payload["nsfw"] = True
        # elif self.censor_nsfw:
        #     payload["censor_nsfw"] = True
        # if not self.trusted_workers:
        #     payload["trusted_workers"] = False
        # if len(self.workers) > 0:
        #     payload["workers"] = self.workers

        # if model != "Random":
        #     payload["models"] = [model]


        response = request_or_error(service, '/v2/generate/async', headers, method='POST', data=payload)
        uuid = response['id']
        
        state.sampling_steps = 0
        state.job_count = n
        start = time.time()

        while True:
            status = request_or_error(service, f'/v2/generate/check/{uuid}', headers)

            elapsed = int(time.time() - start)
            state.sampling_steps = elapsed + status["wait_time"]
            state.sampling_step = elapsed
            # state.current_image_sampling_step = (status['finished']/n)*state.sampling_steps
            
            if status['done']:
                state.sampling_steps = state.sampling_step

                response = request_or_error(service, f'/v2/generate/status/{uuid}', headers)

                images = [get_image(generation['img']) for generation in response['generations']]
                all_seeds=n*[p.seed]
                all_subseeds=n*[p.subseed]
                all_prompts=n*[p.prompt]
                all_negative_prompts=n*[p.negative_prompt]
                infotexts = n*[modules.processing.create_infotext(p, all_prompts, all_seeds, all_subseeds, all_negative_prompts=all_negative_prompts)]
                
                return Processed(
                    p=p, 
                    images_list=images,
                    seed=p.seed,
                    subseed=p.subseed,
                    all_seeds=all_seeds,
                    all_subseeds=all_subseeds,
                    all_prompts=all_prompts,
                    all_negative_prompts=all_negative_prompts,
                    infotexts=infotexts
                    )

    #================================== OmniInfer ==================================
    elif service == RemoteService.OmniInfer:
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            headers = {
                "X-Omni-Key": get_api_key(service),
                "Content-Type": "application/json"
            }
            payload = {
                "prompt": p.prompt,
                "negative_prompt": p.negative_prompt,
                "model_name": modules.shared.sd_model.sd_checkpoint_info.filename,
                "sampler_name": p.sampler_name,
                "batch_size": p.batch_size,
                "n_iter": p.n_iter,
                "steps": p.steps,
                "cfg_scale": p.cfg_scale,
                "clip_skip": p.clip_skip,
                "height": p.height,
                "width": p.width,
                "seed": p.seed
            }
            
            response = request_or_error(service, '/v2/txt2img', headers, method='POST', data=payload)
            if response['code'] != 0:
                raise GenerateRemoteError(service, response['msg'])
            uuid = response['data']['task_id']

            state.sampling_steps = p.steps
            state.job_count = p.n_iter

            processed_image_count = 1
            while True:
                response = request_or_error(service, f'/v2/progress?task_id={uuid}', headers)

                if response['data']['status'] == 1:
                    state.sampling_step = response['data']['progress']*state.sampling_steps
                    current_images = response['data']['current_images']
                    if len(current_images) > processed_image_count:
                        processed_image_count = len(current_images)
                        state.nextjob()
                    last_image_string = next((item for item in reversed(current_images) if item), None)
                    if last_image_string:
                        state.assign_current_image(decode_image(last_image_string))

                elif response['data']['status'] == 2:
                    state.textinfo = "Downloading images..."
                    images = download_images(response['data']['imgs'])
                    info = json.loads(response['data']['info'])
                    return Processed(
                        p=p, 
                        images_list=images,
                        seed=info['seed'],
                        subseed=info['subseed'],
                        all_seeds=info['all_seeds'],
                        all_subseeds=info['all_subseeds'],
                        all_prompts=info['all_prompts'],
                        all_negative_prompts=info['all_negative_prompts'],
                        infotexts=info['infotexts']
                        )
                elif response['data']['status'] == 3:
                    raise GenerateRemoteError(service, 'Generation failed')
                elif response['data']['status'] == 4:
                    raise GenerateRemoteError(service, 'Generation timed out')

        elif isinstance(p, StableDiffusionProcessingImg2Img):
            pass

def process_images(p: StableDiffusionProcessing) -> Processed:
    remote_service = get_current_api_service()

    state.begin()
    state.textinfo = f"Remote inference from {remote_service}"
    p = generate_images(remote_service, p)
    state.end()
    return p