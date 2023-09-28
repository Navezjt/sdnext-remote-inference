import modules.processing
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, Processed
import modules.shared
from modules.shared import state, log

from extension.utils_remote import get_current_api_service, request_or_error, RemoteService, get_api_key

from multiprocessing.pool import ThreadPool
import requests
import base64
import json
import io
from PIL import Image

class GenerateRemoteError(Exception):
    def __init__(self, service, error):
        super().__init__(f'RI: Unable to remotely infer task for {service}: {error}')

def generate_images(service: RemoteService, p: StableDiffusionProcessing) -> Processed:
    if service == RemoteService.SDNext:
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            pass
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            pass

    elif service ==  RemoteService.StableHorde:
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            pass
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            pass

    elif service == RemoteService.OmniInfer:
        if isinstance(p, StableDiffusionProcessingTxt2Img):
            headers = {
                "X-Omni-Key": get_api_key(service),
                "Content-Type": "application/json"
            }
            p.seed = int(modules.processing.get_fixed_seed(p.seed))
            data = {
                "prompt": p.prompt,
                "negative_prompt": p.negative_prompt,
                "model_name": modules.shared.sd_model.sd_checkpoint_info.filename,
                "sampler_name": p.sampler_name,
                "batch_size": p.batch_size,
                "n_iter": p.n_iter,
                "steps": p.steps,
                "cfg_scale": p.cfg_scale,
                "height": p.height,
                "width": p.width,
                "seed": p.seed
            }
            
            response = request_or_error(service, '/v2/txt2img', headers, method='POST', data=data)
            if response['code'] != 0:
                raise GenerateRemoteError(service, response['msg'])
            uuid = response['data']['task_id']
            state.nextjob()

            processed_image_count = 1
            while True:
                response = request_or_error(service, f'/v2/progress?task_id={uuid}', headers)

                if response['data']['status'] == 1:
                    state.current_image_sampling_step = response['data']['progress']*state.sampling_steps
                    current_images = response['data']['current_images']
                    if len(current_images) > processed_image_count:
                        processed_image_count = len(current_images)
                        state.nextjob()
                    
                    last_image_string = next((item for item in reversed(current_images) if item), None)
                    if last_image_string:
                        state.assign_current_image(Image.open(io.BytesIO(base64.b64decode(last_image_string))))

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

def download_images(img_urls, num_threads=10):
    # Copyright OmniInfer
    def _download(img_url):
        attempts = 5
        while attempts > 0:
            try:
                response = requests.get(img_url, timeout=5)
                response.raise_for_status()
                with io.BytesIO(response.content) as fp:
                    return Image.open(fp).copy()
            except (requests.RequestException, Image.UnidentifiedImageError):
                log.warning(f"RI: Failed to download {img_url}, retrying...")
                attempts -= 1
        return None

    with ThreadPool(num_threads) as pool:
        images = pool.map(_download, img_urls)

    return [img for img in images if img is not None]

def process_images(p: StableDiffusionProcessing) -> Processed:
    log.debug("test")

    remote_service = get_current_api_service()

    state.begin()
    state.sampling_steps = p.steps
    state.job_count = p.n_iter
    state.textinfo = f"Remote inference from {remote_service}"

    p = generate_images(remote_service, p)
    state.end()
    return p