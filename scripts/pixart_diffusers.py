import math
import torch
import gc
import json
import numpy as np
import os

from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.rng import create_generator
from modules.shared import opts
from modules.ui_components import ResizeHandleRow, ToolButton
import modules.infotext_utils as parameters_copypaste
import gradio as gr

from PIL import Image
#workaround for unnecessary flash_attn requirement for Florence-2
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoProcessor, AutoModelForCausalLM 

#torch.backends.cuda.enable_flash_sdp(True)
#torch.backends.cuda.enable_mem_efficient_sdp(False)     #   minimal difference

import customStylesListPA as styles
import modelsListPA as models

class PixArtStorage:
    lastSeed = -1
    galleryIndex = 0
    lastPrompt = None
    lastNegative = None
    pos_embeds = None
    pos_attention = None
    neg_embeds = None
    neg_attention = None
    denoise = 0.0
    karras = False
    resolutionBin = True
    noiseRGBA = [0.0, 0.0, 0.0, 0.0]
    captionToPrompt = False


from transformers import T5EncoderModel, T5Tokenizer
from diffusers import PixArtSigmaPipeline, PixArtAlphaPipeline, PixArtTransformer2DModel
from diffusers import AutoencoderKL
from diffusers import ConsistencyDecoderVAE
from diffusers import DEISMultistepScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, UniPCMultistepScheduler, DDPMScheduler
from diffusers import SASolverScheduler, LCMScheduler
from peft import PeftModel

from diffusers.utils.torch_utils import randn_tensor


import argparse
import pathlib
from pathlib import Path
import sys

current_file_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_file_path))



# modules/infotext_utils.py
def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text

    return json.dumps(text, ensure_ascii=False)

# modules/processing.py
def create_infotext(model, positive_prompt, negative_prompt, guidance_scale, steps, DMDstep, seed, scheduler, width, height):
    karras = " : Karras" if PixArtStorage.karras == True else ""
    isDMD = "PixArt-Alpha-DMD" in model
    generation_params = {
        "Size": f"{width}x{height}",
        "Seed": seed,
        "Scheduler": f"{scheduler}{karras}",
        "Steps": steps if not isDMD else None,
        "CFG": guidance_scale if not isDMD else None,
        "DMDstep": DMDstep if isDMD else None,
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None
    }

#add i2i marker - effectively just check PixArtStorage.denoise as if =1, no i2i effect

    prompt_text = f"Prompt: {positive_prompt}\n"
    if negative_prompt != "":
        prompt_text += (f"Negative: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])
    noise_text = f"\nInitial noise: {PixArtStorage.noiseRGBA}" if PixArtStorage.noiseRGBA[3] != 0.0 else ""

    return f"Model: {model}\n{prompt_text}{generation_params_text}{noise_text}"

def predict(positive_prompt, negative_prompt, model, vae, width, height, guidance_scale, num_steps, DMDstep, sampling_seed, num_images, scheduler, i2iSource, i2iDenoise, style, *args):

    torch.set_grad_enabled(False)

    if style != 0:
        positive_prompt = styles.styles_list[style][1].replace("{prompt}", positive_prompt)
        negative_prompt = styles.styles_list[style][2] + negative_prompt

    if i2iSource == None:
        i2iDenoise = 1
    if i2iDenoise < (num_steps + 1) / 1000:
        i2iDenoise = (num_steps + 1) / 1000

    from diffusers.utils import logging
    logging.set_verbosity(logging.WARN)       #   download information is useful

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(sampling_seed)
    PixArtStorage.lastSeed = fixed_seed

    ####    identify model type basde on name
    isFlash = "flash-pixart" in model
    isSigma = "PixArt-Sigma" in model
    isDMD = "PixArt-Alpha-DMD" in model
    isLCM = "PixArt-LCM" in model

    if model in models.models_list_alpha:
        isSigma = False
    if model in models.models_list_sigma:
        isSigma = True

    useConsistencyVAE = (isSigma == 0) and (vae == 1)

#    algorithm_type = args.algorithm
#    beta_schedule = args.beta_schedule
#    use_lu_lambdas = args.use_lu_lambdas

    pipe = None

    useCachedEmbeds = (PixArtStorage.lastPrompt == positive_prompt and PixArtStorage.lastNegative == negative_prompt)

    if useCachedEmbeds:
        print ("Skipping tokenizer, text_encoder.")
        tokenizer = None
        text_encoder = None
    else:
        ####    tokenizer always the same
        tokenizer = T5Tokenizer.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder="tokenizer", )

        ####    the T5 text encoder model is always the same, so it's easy to cache and share between PixArt models
        try:
            text_encoder = T5EncoderModel.from_pretrained(
                ".//models//diffusers//pixart_T5_fp16",
                variant="fp16",
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="auto", )
        except:
        ##  fetch the T5 model, ~20gigs, load as fp16
        ##  specifying cache directory because transformers will cache to a different location than diffusers, so you can have 20gigs of T5 twice
            text_encoder = T5EncoderModel.from_pretrained(
                "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder="text_encoder",
                use_safetensors=True,
                torch_dtype=torch.float16, )

            ##  now save the converted fp16 T5 model to local cache, only needs done once
            text_encoder.to(torch.float16)
            text_encoder.save_pretrained(
                ".//models//diffusers//pixart_T5_fp16",
                variant="fp16",
                safe_serialization=True, )
            print ("Saved fp16 T5 text encoder, will use this from now on.")

##        try:
##            from optimum.bettertransformer import BetterTransformer
##            text_encoder = BetterTransformer.transform(text_encoder)
##        except:
##            print ("BetterTransformer not available.")

####    VAEs are same for Alpha, and for Sigma. Sigma already shared, now Alpha is too.

    if useConsistencyVAE:   #   option for Alpha models
        vae = ConsistencyDecoderVAE.from_pretrained(
            "openai/consistency-decoder",
            local_files_only=False, cache_dir=".//models//diffusers//",
            torch_dtype=torch.float16)
    else:    
        cachedVAE = ".//models//diffusers//pixart_T5_fp16//vaeSigma" if isSigma else ".//models//diffusers//pixart_T5_fp16//vaeAlpha"
        sourceVAE = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers" if isSigma else model

        try:
            vae = AutoencoderKL.from_pretrained(cachedVAE, variant="fp16", torch_dtype=torch.float16)
        except:
            vae = AutoencoderKL.from_pretrained(
                sourceVAE,
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder="vae",
                use_safetensors=True,
                torch_dtype=torch.float16, )

            ##  now save the converted fp16 T5 model to local cache, only needs done once
            vae.to(torch.float16)
            vae.save_pretrained(
                cachedVAE,
                safe_serialization=True,
                variant="fp16", )
            print ("Saved fp16 " + "Sigma" if isSigma else "Alpha" + " VAE, will use this from now on.")

    vae.enable_tiling(True) #make optional/based on dimensions

    logging.set_verbosity(logging.ERROR)       #   avoid some console spam from Alpha models missing keys


    if isFlash:
        pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=None,
            vae=vae,
            torch_dtype=torch.float16, )        
    elif isSigma:
        pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
            local_files_only=False, cache_dir=".//models//diffusers//",
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=None,
            vae=vae,
            torch_dtype=torch.float16, )
    else:
        pipe = PixArtAlphaPipeline.from_pretrained(
            model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=None,
            vae=vae,
            torch_dtype=torch.float16, )

    del vae

    if isDMD:
        negative_prompt = ""

    if useCachedEmbeds == False:
        pos_embeds, pos_attention, neg_embeds, neg_attention = pipe.encode_prompt(positive_prompt, negative_prompt=negative_prompt)

        pipe.tokenizer = None
        pipe.text_encoder = None
        del tokenizer, text_encoder

        PixArtStorage.pos_embeds    = pos_embeds.to('cuda').to(torch.float16)
        PixArtStorage.neg_embeds    = neg_embeds.to('cuda').to(torch.float16)
        PixArtStorage.pos_attention = pos_attention.to('cuda').to(torch.float16)
        PixArtStorage.neg_attention = neg_attention.to('cuda').to(torch.float16)

        PixArtStorage.lastPrompt = positive_prompt
        PixArtStorage.lastNegative = negative_prompt

    gc.collect()
    torch.cuda.empty_cache()


####    load transformer, same process for Alpha and Sigma
    if isFlash: #   base is an attempt at future proofing - base flash pipeline is currently always Alpha
        base = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS" if isSigma else "PixArt-alpha/PixArt-XL-2-1024-MS"
        # Load LoRA
        transformer = PixArtTransformer2DModel.from_pretrained(
            base,
            subfolder="transformer",
            torch_dtype=torch.float16
        )
        transformer = PeftModel.from_pretrained(
            transformer,
            "jasperai/flash-pixart"
        )
    else:
        transformer = PixArtTransformer2DModel.from_pretrained(
            model,
            local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder='transformer',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
            device_map=None, )

    pipe.transformer = transformer
    del transformer

##    # LoRA model -can't find examples in necessary form
##    loraLocation = ".//models//diffusers//PixArtLora"
##    loraName = "Wednesday.safetensors"
##    transformer = PeftModel.from_single_file(
##        transformer,
##        loraLocation,
##        adapter_name=loraName,
##        config=None,
##        local_files_only=True)


    pipe.to('cuda')
#    pipe.enable_model_cpu_offload()

    #   if using resolution_binning, must use adjusted width/height here (don't overwrite values)
    #   always generate the noise here
    generator = [torch.Generator(device='cpu').manual_seed(fixed_seed+i) for i in range(num_images)]

    if PixArtStorage.resolutionBin:
        from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
            ASPECT_RATIO_256_BIN,
            ASPECT_RATIO_512_BIN,
            ASPECT_RATIO_1024_BIN,
        )
        from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import (
            ASPECT_RATIO_2048_BIN,
        )

        if pipe.transformer.config.sample_size == 256:
            aspect_ratio_bin = ASPECT_RATIO_2048_BIN
        elif pipe.transformer.config.sample_size == 128:
            aspect_ratio_bin = ASPECT_RATIO_1024_BIN
        elif pipe.transformer.config.sample_size == 64:
            aspect_ratio_bin = ASPECT_RATIO_512_BIN
        elif pipe.transformer.config.sample_size == 32:
            aspect_ratio_bin = ASPECT_RATIO_256_BIN
        else:
            raise ValueError("Invalid sample size")

        ar = float(height / width)
        closest_ratio = min(aspect_ratio_bin.keys(), key=lambda ratio: abs(float(ratio) - ar))
        theight = int(aspect_ratio_bin[closest_ratio][0])
        twidth  = int(aspect_ratio_bin[closest_ratio][1])
    else:
        theight = height
        twidth  = width

    shape = (
        num_images,
        pipe.transformer.config.in_channels,
        int(theight) // pipe.vae_scale_factor,
        int(twidth) // pipe.vae_scale_factor,
    )

    latents = randn_tensor(shape, generator=generator, dtype=torch.float16).to('cuda').to(torch.float16)

    #   colour the initial noise
    if PixArtStorage.noiseRGBA[3] != 0.0:
        nr = PixArtStorage.noiseRGBA[0] ** 0.5
        ng = PixArtStorage.noiseRGBA[1] ** 0.5
        nb = PixArtStorage.noiseRGBA[2] ** 0.5

        imageR = torch.tensor(np.full((8,8), (nr), dtype=np.float32))
        imageG = torch.tensor(np.full((8,8), (ng), dtype=np.float32))
        imageB = torch.tensor(np.full((8,8), (nb), dtype=np.float32))
        image = torch.stack((imageR, imageG, imageB), dim=0).unsqueeze(0)

        image = pipe.image_processor.preprocess(image).to('cuda').to(torch.float16)
        image_latents = pipe.vae.encode(image).latent_dist.sample(generator) * pipe.vae.config.scaling_factor * pipe.scheduler.init_noise_sigma
        image_latents = image_latents.repeat(num_images, 1, latents.size(2), latents.size(3))

        for b in range(len(latents)):
            for c in range(4):
                latents[b][c] -= latents[b][c].mean()

#        latents += image_latents * PixArtStorage.noiseRGBA[3]
#        torch.lerp (latents, image_latents, PixArtStorage.noiseRGBA[3], out=latents)
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        ts = torch.tensor([int(1000 * (1.0-PixArtStorage.noiseRGBA[3])) - 1], device='cpu')
        ts = ts[:1].repeat(num_images)

        latents = pipe.scheduler.add_noise(image_latents, latents, ts)

        del imageR, imageG, imageB, image, image_latents
    #   end: colour the initial noise


    if i2iSource != None:
        i2iSource = i2iSource.resize((twidth, theight))

        image = pipe.image_processor.preprocess(i2iSource).to('cuda').to(torch.float16)
        image_latents = pipe.vae.encode(image).latent_dist.sample(generator) * pipe.vae.config.scaling_factor * pipe.scheduler.init_noise_sigma
        image_latents = image_latents.repeat(num_images, 1, 1, 1)

        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        ts = torch.tensor([int(1000 * i2iDenoise) - 1], device='cpu')
        ts = ts[:1].repeat(num_images)

        latents = pipe.scheduler.add_noise(image_latents, latents, ts)

        del image, image_latents, i2iSource

    timesteps = None

#    if useCustomTimeSteps:
#    timesteps = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13]    #   AYS sdXL
    #loglin interpolate to number of steps
    
    if isDMD:
        guidance_scale = 1
        num_steps = 1
        timesteps = [DMDstep]

    if isDMD or isLCM:
        scheduler = 'default'
    elif isFlash:
        # Scheduler
        pipe.scheduler = LCMScheduler.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            subfolder="scheduler",
            timestep_spacing="trailing",
        )
        scheduler = 'LCM'
    elif scheduler == 'DDPM':
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DEIS':
        pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DPM++ 2M':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == "DPM++ 2M SDE":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type='sde-dpmsolver++')
    elif scheduler == 'DPM':
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DPM SDE':
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Euler':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Euler A':
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == "SA-solver":
        pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config, algorithm_type='data_prediction')
    elif scheduler == 'UniPC':
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#   else uses default set by model

    pipe.scheduler.config.num_train_timesteps = int(1000 * i2iDenoise)

    if hasattr(pipe.scheduler.config, 'use_karras_sigmas'):
        pipe.scheduler.config.use_karras_sigmas = PixArtStorage.karras


##    pipe.scheduler.beta_schedule  = beta_schedule
##    pipe.scheduler.use_lu_lambdas = use_lu_lambdas
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        output = pipe(
            latents=latents,
            negative_prompt=None, 
            num_inference_steps=num_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            prompt_embeds=PixArtStorage.pos_embeds,
            negative_prompt_embeds=PixArtStorage.neg_embeds,
            prompt_attention_mask=PixArtStorage.pos_attention,
            negative_prompt_attention_mask=PixArtStorage.neg_attention,
            num_images_per_prompt=num_images,
            output_type="pil",
            generator=generator,
            use_resolution_binning=PixArtStorage.resolutionBin,
            timesteps=timesteps,
        ).images

#   vae uses lots of VRAM, especially with batches, maybe worth output to latent, free memory
#   but seem to need vae loaded earlier anyway

    del pipe.transformer, generator, latents
    pipe.transformer = None
    gc.collect()
    torch.cuda.empty_cache()

    result = []
    for image in output:
        info=create_infotext(
            model,
            positive_prompt, negative_prompt,
            guidance_scale, num_steps, DMDstep,
            fixed_seed, scheduler,
            width, height, )

        result.append((image, info))
        
        images.save_image(
            image,
            opts.outdir_samples or opts.outdir_txt2img_samples,
            "",
            fixed_seed,
            positive_prompt,
            opts.samples_format,
            info
        )
        fixed_seed += 1

    del output, pipe
    gc.collect()
    torch.cuda.empty_cache()

    return gr.Button.update(value='Generate', variant='primary', interactive=True), result



def on_ui_tabs():

    models_list = models.models_list_alpha + models.models_list_sigma
    defaultModel = models.defaultModel
    defaultWidth = models.defaultWidth
    defaultHeight = models.defaultHeight
    
    def getGalleryIndex (evt: gr.SelectData):
        PixArtStorage.galleryIndex = evt.index

    def reuseLastSeed ():
        return PixArtStorage.lastSeed + PixArtStorage.galleryIndex
        
    def randomSeed ():
        return -1

    def i2iSetDimensions (image, w, h):
        if image is not None:
            w = image.size[0]
            h = image.size[1]
        return [w, h]

#add a blur?


    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
        return imports
    def i2iMakeCaptions (image, originalPrompt):
        if image == None:
            return originalPrompt

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', 
                                                         attn_implementation="sdpa", 
                                                         torch_dtype=torch.float32, 
                                                         cache_dir=".//models//diffusers//", 
                                                         trust_remote_code=True)
        processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', #-large
                                                  torch_dtype=torch.float32, 
                                                  cache_dir=".//models//diffusers//", 
                                                  trust_remote_code=True)

        result = ''
        prompts = ['<MORE_DETAILED_CAPTION>']

        for p in prompts:
            inputs = processor(text=p, images=image, return_tensors="pt")
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            del inputs
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            del generated_ids
            parsed_answer = processor.post_process_generation(generated_text, task=p, image_size=(image.width, image.height))
            del generated_text
            print (parsed_answer)
            result += parsed_answer[p]
            if p != prompts[-1]:
                result += ' | \n'
            del parsed_answer

        del model, processor

        if PixArtStorage.captionToPrompt:
            return result
        else:
            return originalPrompt



    def i2iImageFromGallery (gallery):
        try:
            newImage = gallery[PixArtStorage.galleryIndex][0]['name'].split('?')
            return newImage[0]
        except:
            return None
    def toggleC2P ():
        PixArtStorage.captionToPrompt ^= True
        return gr.Button.update(variant=['secondary', 'primary'][PixArtStorage.captionToPrompt])

    def toggleKarras ():
        PixArtStorage.karras ^= True
        return gr.Button.update(variant='primary' if PixArtStorage.karras == True else 'secondary',
                                value='\U0001D40A' if PixArtStorage.karras == True else '\U0001D542')
    def toggleResBin ():
        PixArtStorage.resolutionBin ^= True
        return gr.Button.update(variant='primary' if PixArtStorage.resolutionBin == True else 'secondary',
                                value='\U0001D401' if PixArtStorage.resolutionBin == True else '\U0001D539')

    def toggleGenerate (R, G, B, A):
        PixArtStorage.noiseRGBA = [R, G, B, A]
        return gr.Button.update(value='...', variant='secondary', interactive=False)

    with gr.Blocks() as pixartsigma2_block:
        with ResizeHandleRow():
            with gr.Column():
                with gr.Row():
                    model = gr.Dropdown(models_list, label='Model', value=defaultModel, type='value', scale=2)
                    vae = gr.Dropdown(["default", "consistency"], label='VAE', value="default", type='index', scale=1)
                    scheduler = gr.Dropdown(["default",
                                             "DDPM",
                                             "DEIS",
                                             "DPM++ 2M",
                                             "DPM++ 2M SDE",
                                             "DPM",
                                             "DPM SDE",
                                             "Euler",
                                             "Euler A",
                                             "SA-solver",
                                             "UniPC",
                                             ],
                        label='Sampler', value="UniPC", type='value', scale=1)
                    karras = ToolButton(value="\U0001D542", variant='secondary', tooltip="use Karras sigmas")

                positive_prompt = gr.Textbox(label='Prompt', placeholder='Enter a prompt here...', default='', lines=2)

                with gr.Row():
                    negative_prompt = gr.Textbox(label='Negative', placeholder='', lines=2)
                    style = gr.Dropdown([x[0] for x in styles.styles_list], label='Style', value="(None)", type='index', scale=0)
                with gr.Row():
                    width = gr.Slider(label='Width', minimum=128, maximum=4096, step=8, value=defaultWidth, elem_id="PixArtSigma_width")
                    swapper = ToolButton(value="\U000021C4")
                    height = gr.Slider(label='Height', minimum=128, maximum=4096, step=8, value=defaultHeight, elem_id="PixArtSigma_height")
                    resBin = ToolButton(value="\U0001D401", variant='primary', tooltip="use resolution binning")

                with gr.Row():
                    guidance_scale = gr.Slider(label='CFG', minimum=1, maximum=8, step=0.5, value=4.0, scale=2, visible=True)
                    steps = gr.Slider(label='Steps', minimum=1, maximum=60, step=1, value=20, scale=2, visible=True)
                    DMDstep = gr.Slider(label='Timestep for DMD', minimum=1, maximum=999, step=1, value=400, scale=1, visible=False)
                with gr.Row():
                    sampling_seed = gr.Number(label='Seed', value=-1, precision=0, scale=1)
                    random = ToolButton(value="\U0001f3b2\ufe0f")
                    reuseSeed = ToolButton(value="\u267b\ufe0f")
                    batch_size = gr.Number(label='Batch Size', minimum=1, maximum=9, value=1, precision=0, scale=0)

                with gr.Accordion(label='the colour of noise', open=False):
                    with gr.Row():
                        initialNoiseR = gr.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='red')
                        initialNoiseG = gr.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='green')
                        initialNoiseB = gr.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='blue')
                        initialNoiseA = gr.Slider(minimum=0, maximum=0.3, value=0.0, step=0.005, label='strength')

                with gr.Accordion(label='image to image', open=False):
                    with gr.Row():
                        i2iSource = gr.Image(label='image to image source', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        with gr.Column():
                            i2iDenoise = gr.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                            i2iSetWH = gr.Button(value='Set Width / Height from image')
                            i2iFromGallery = gr.Button(value='Get image from gallery')
                            with gr.Row():
                                i2iCaption = gr.Button(value='Caption this image (Florence-2)', scale=9)
                                toPrompt = ToolButton(value='P', variant='secondary')

                ctrls = [positive_prompt, negative_prompt, model, vae, width, height, guidance_scale, steps, DMDstep, sampling_seed, batch_size, scheduler, i2iSource, i2iDenoise, style]

            with gr.Column():
                generate_button = gr.Button(value="Generate", variant='primary', visible=True)
                output_gallery = gr.Gallery(label='Output', height=shared.opts.gallery_height or None,
                                            show_label=False, object_fit='contain', visible=True, columns=3, preview=True)
#   gallery movement buttons don't work, others do
#   caption not displaying linebreaks, alt text does

                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname,
                        source_text_component=positive_prompt,
                        source_image_component=output_gallery,
                    ))


        def show_steps(m):
            if "PixArt-Alpha-DMD" in m:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

        model.change(
            fn=show_steps,
            inputs=model,
            outputs=[guidance_scale, steps, DMDstep],
            show_progress=False
        )


        karras.click(toggleKarras, inputs=[], outputs=karras)
        resBin.click(toggleResBin, inputs=[], outputs=resBin)
        swapper.click(fn=None, _js="function(){switchWidthHeight('PixArtSigma')}", inputs=None, outputs=None, show_progress=False)
        random.click(randomSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        reuseSeed.click(reuseLastSeed, inputs=[], outputs=sampling_seed, show_progress=False)

        i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource, width, height], outputs=[width, height], show_progress=False)
        i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery], outputs=[i2iSource])
        i2iCaption.click (fn=i2iMakeCaptions, inputs=[i2iSource, positive_prompt], outputs=[positive_prompt])#outputs=[positive_prompt]
        toPrompt.click(toggleC2P, inputs=[], outputs=[toPrompt])

        output_gallery.select (fn=getGalleryIndex, inputs=[], outputs=[])

        generate_button.click(toggleGenerate, inputs=[initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA], outputs=[generate_button])
        generate_button.click(predict, inputs=ctrls, outputs=[generate_button, output_gallery])

    return [(pixartsigma2_block, "PixArtSigma", "pixart_sigma")]

script_callbacks.on_ui_tabs(on_ui_tabs)

