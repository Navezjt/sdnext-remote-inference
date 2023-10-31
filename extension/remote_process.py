import modules.processing
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, Processed
import modules.shared
from modules.shared import state, log, opts
import modules.images

from extension.utils_remote import encode_image, decode_image, download_images, get_current_api_service, get_image, request_or_error, RemoteService, get_api_key, stable_horde_samplers, stable_horde_client

import json
import time
import math
from PIL import Image
import re

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
        data = vars(p)
        txt2img_keys = ["enable_hr", "denoising_strength", "firstphase_width", "firstphase_height", "hr_scale", "hr_force", "hr_upscaler", "hr_second_pass_steps", "hr_resize_x", "hr_resize_y", "refiner_steps", "refiner_start", "refiner_prompt", "refiner_negative", "prompt", "styles", "seed", "subseed", "subseed_strength", "seed_resize_from_h", "seed_resize_from_w", "sampler_name", "latent_sampler", "batch_size", "n_iter", "steps", "cfg_scale", "image_cfg_scale", "clip_skip", "width", "height", "full_quality", "restore_faces", "tiling", "do_not_save_samples", "do_not_save_grid", "negative_prompt", "eta", "diffusers_guidance_rescale", "override_settings", "override_settings_restore_afterwards", "sampler_index", "script_name", "send_images", "save_images", "alwayson_scripts"] #"script_args", "sampler_index"
        img2img_keys = ["init_images", "resize_mode", "denoising_strength", "image_cfg_scale", "mask", "mask_blur", "inpainting_fill", "inpaint_full_res", "inpaint_full_res_padding", "inpainting_mask_invert", "initial_noise_multiplier", "refiner_steps", "refiner_start", "refiner_prompt", "refiner_negative", "prompt", "styles", "seed", "subseed", "subseed_strength", "seed_resize_from_h", "seed_resize_from_w", "sampler_name", "latent_sampler", "batch_size", "n_iter", "steps", "cfg_scale", "clip_skip", "width", "height", "full_quality", "restore_faces", "tiling", "do_not_save_samples", "do_not_save_grid", "negative_prompt", "eta", "diffusers_guidance_rescale", "override_settings", "override_settings_restore_afterwards", "sampler_index", "include_init_images", "script_name", "send_images", "save_images", "alwayson_scripts"] #"script_args", "sampler_index"
        processed_keys = ["seed", "info", "subseed", "all_prompts", "all_negative_prompts", "all_seeds", "all_subseeds", "index_of_first_image", "infotexts", "comments"]

        request_or_error(service, '/sdapi/v1/options', method='POST', data={'sd_model_checkpoint': opts.sd_model_checkpoint})
        request_or_error(service, '/sdapi/v1/reload-checkpoint', method='POST')

        if isinstance(p, StableDiffusionProcessingTxt2Img): 
            data = {key: data[key] for key in txt2img_keys if key in data.keys()}
            log.debug(data)
            response = request_or_error(service, '/sdapi/v1/txt2img', method='POST', data=data)
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            data = {key: data[key] for key in img2img_keys if key in data.keys() and data[key] is not None}
            log.debug(data)
            data['init_images'] = [encode_image(img) for img in data['init_images']]
            response = request_or_error(service, '/sdapi/v1/img2img', method='POST', data=data)

        log.debug(f"RI: response: {response}")
        info = json.loads(response['info'])
        info = {key: info[key] for key in processed_keys if key in info.keys()}
        return Processed(p=p, images_list=[decode_image(img) for img in response['images']], **info)

    #================================== Stable Horde ==================================
    elif service ==  RemoteService.StableHorde:
    # Copyright NatanJunges https://github.com/natanjunges/stable-diffusion-webui-stable-horde/blob/00248b89bfab7ba465f104324a5d0708ad37341f/scripts/main.py#L376
        n = p.n_iter*p.batch_size

        def extract(instr, pattern):
            return re.sub(pattern, '', instr), re.findall(pattern, instr)
        
        prompt = p.prompt
        negative_prompt = p.negative_prompt

        prompt, pos_loras = extract(prompt, r'<lora:(\w+):([\d.]+)>')
        negative_prompt, neg_loras = extract(negative_prompt, r'<lora:(\w+):([\d.]+)>')
        loras = [(i,float(j)) for i,j in pos_loras] + [(i,-float(j)) for i,j in neg_loras]
        tis = re.findall(r'embedding:(\w+)', prompt+negative_prompt)

        if len(loras) > 5:
            raise GenerateRemoteError(service, 'Too many loras')


        headers = {
            "apikey": get_api_key(service),
            "Client-Agent": stable_horde_client,
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": f"{prompt}###{negative_prompt}" if negative_prompt else prompt,
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
                "n": n,
                "loras": [{"name": i, "model": j} for i,j in loras],
                "tis": [{"name": i} for i in tis]
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

        response = request_or_error(service, '/v2/generate/async', headers, method='POST', data=payload)
        uuid = response['id']
        
        state.sampling_steps = 100
        state.job_count = n
        start = time.time()

        while True:
            status = request_or_error(service, f'/v2/generate/check/{uuid}', headers)

            elapsed = int(time.time() - start)
            state.sampling_steps = elapsed + status["wait_time"]
            state.sampling_step = elapsed
            
            if status['done']:
                state.sampling_step = state.sampling_steps

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

        if isinstance(p, StableDiffusionProcessingTxt2Img):
            if p.enable_hr:
                payload.update({
                    "enable_hr": True,
                    "hr_upscaler": p.hr_upscaler,
                    "hr_scale": p.hr_scale,
                    "hr_resize_x": p.hr_resize_x,
                    "hr_resize_y": p.hr_resize_y
                })
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            payload.update({
                "init_images": [encode_image(img) for img in p.init_images],
                "denoising_strength": p.denoising_strength,
                "resize_mode": p.resize_mode,
                "image_cfg_scale": p.image_cfg_scale,
                "mask_blur": p.mask_blur,
                "inpainting_fill": p.inpainting_fill,
                "inpaint_full_res": p.inpaint_full_res,
                "inpaint_full_res_padding": p.inpaint_full_res_padding,
                "inpainting_mask_invert": p.inpainting_mask_invert,
                "initial_noise_multiplier": p.initial_noise_multiplier
            })
            if p.image_mask:
                payload["mask"] = [encode_image(p.image_mask)]

        log.debug(f'RI: payload: {payload}')
        
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            response = request_or_error(service, '/v2/txt2img', headers, method='POST', data=payload)
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            response = request_or_error(service, '/v2/img2img', headers, method='POST', data=payload)

        if response['code'] != 0:
            raise GenerateRemoteError(service, response['msg'])
        uuid = response['data']['task_id']

        state.sampling_steps = p.steps
        state.job_count = p.n_iter
        start = time.time()

        while True:
            response = request_or_error(service, f'/v2/progress?task_id={uuid}', headers)

            if response['data']['status'] == 1:
                elapsed = int(time.time() - start)
                state.sampling_steps = elapsed + response['data']['eta_relative']
                state.sampling_step = elapsed

                current_images = response['data']['current_images']
                last_image_string = next((item for item in reversed(current_images) if item), None)
                if last_image_string:
                    state.assign_current_image(decode_image(last_image_string))

            elif response['data']['status'] == 2:
                state.step = state.sampling_steps
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

def save_images_and_add_grid(proc: Processed, p:StableDiffusionProcessing):
    #Copyright NatanJunges https://github.com/natanjunges/stable-diffusion-webui-stable-horde/blob/00248b89bfab7ba465f104324a5d0708ad37341f/scripts/main.py#L345C6-L363C1

    if opts.save and not p.do_not_save_samples:
        for i,img in enumerate(proc.images):
            modules.images.save_image(img, path=p.outpath_samples, basename="", seed=proc.all_seeds[i], prompt=proc.all_prompts[i], extension=opts.samples_format, info=proc.infotexts[i], p=p)

    if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and len(proc.images) >= 2:
        grid = modules.images.image_grid(proc.images, rows=math.ceil(math.sqrt(len(proc.images))))
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