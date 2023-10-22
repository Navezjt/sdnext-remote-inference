import modules.processing
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, Processed
import modules.shared
from modules.shared import state, log, opts
import modules.images

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
        opts.sd_model_checkpoint = checkpoint_info.title
        return RemoteModel(checkpoint_info)
    except StopIteration:
        return None

class GenerateRemoteError(Exception):
    def __init__(self, service, error):
        super().__init__(f'RI: Unable to remotely infer task for {service}: {error}')

def generate_images(service: RemoteService, p: StableDiffusionProcessing) -> Processed:
    p.seed = int(modules.processing.get_fixed_seed(p.seed))
    p.subseed = int(modules.processing.get_fixed_seed(p.subseed))
    p.prompt = modules.shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = modules.shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)

    #================================== SD.Next ==================================
    if service == RemoteService.SDNext:
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            pass
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            pass

    #================================== Stable Horde ==================================
    # Copyright NatanJunges https://github.com/natanjunges/stable-diffusion-webui-stable-horde/blob/00248b89bfab7ba465f104324a5d0708ad37341f/scripts/main.py#L376
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
                "karras": opts.schedulers_sigma == 'karras',
                "cfg_scale": p.cfg_scale,
                "clip_skip": p.clip_skip,
                "seed": str(p.seed),
                "seed_variation": 1000,
                "height": p.height,
                "width": p.width,
                "steps": p.steps,
                "n": n
            },
            "models": [modules.shared.sd_model.sd_checkpoint_info.filename],
            "nsfw": opts.horde_nsfw,
            "trusted_workers": opts.horde_trusted_workers,
            "slow_workers": opts.horde_slow_workers,
            "censor_nsfw": opts.horde_censor_nsfw,
            "workers": opts.horde_workers.split(',') if len(opts.horde_workers) > 0 else [],
            "worker_blacklist": opts.horde_worker_blacklist,
            "shared": opts.horde_share_laion
        }

        if isinstance(p, StableDiffusionProcessingTxt2Img):
            payload["params"]["tiling"] = p.tiling
            payload["params"]["hires_fix"] = p.enable_hr
            if p.enable_hr:
                payload["params"]["denoising_strength"] = p.denoising_strength
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            payload["source_image"] = encode_image(p.init_images[0])
            payload["params"]["denoising_strength"] = p.denoising_strength

            if p.image_mask:
                payload["source_processing"] = "inpainting"
                payload["source_mask"] = encode_image(p.image_mask)
        else:
            return

        log.debug(f'RI: payload: {payload}')
        
        # todo
        # controlnet, postprocessing, upscale
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
                all_seeds=[p.seed + i*1000 for i in range(n)]
                all_subseeds=n*[1000]
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
            elif status['faulted']:
                raise GenerateRemoteError(service, 'Generation failed')
            elif not status['is_possible']:
                raise GenerateRemoteError(service, 'Generation not possible with current worker pool')


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

def save_images_and_add_grid(proc: Processed, p:StableDiffusionProcessing):
    #Copyright NatanJunges https://github.com/natanjunges/stable-diffusion-webui-stable-horde/blob/00248b89bfab7ba465f104324a5d0708ad37341f/scripts/main.py#L345C6-L363C1

    if opts.save and not p.do_not_save_samples:
        for i,img in enumerate(proc.images):
            modules.images.save_image(img, path=p.outpath_samples, basename="", seed=proc.all_seeds[i], prompt=proc.all_prompts[i], extension=opts.samples_format, info=proc.infotexts[i], p=p)

    if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not (len(proc.images) < 2 and opts.grid_only_if_multiple):
        grid = modules.images.image_grid(proc.images, len(proc.images))
        info = '\n'.join(proc.infotexts)

        if opts.return_grid:
            proc.infotexts.insert(0, info)
            proc.images.insert(0, grid)
            proc.index_of_first_image = 1

        if opts.grid_save:
            modules.images.save_image(grid, p.outpath_grids, "grid", proc.all_seeds[0], proc.all_prompts[0], opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, p=p, grid=True)

    return proc

def process_images(p: StableDiffusionProcessing) -> Processed:
    remote_service = get_current_api_service()

    state.begin()
    state.textinfo = f"Remote inference from {remote_service}"
    proc = generate_images(remote_service, p)
    proc = save_images_and_add_grid(proc, p)
    state.end()
    return proc