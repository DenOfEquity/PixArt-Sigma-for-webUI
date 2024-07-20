import math
import torch
import gc
import json
import numpy as np
import os

from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.shared import opts
from modules.ui_components import ResizeHandleRow, ToolButton
import modules.infotext_utils as parameters_copypaste
import gradio as gr

#workaround for unnecessary flash_attn requirement for Florence-2
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoProcessor, AutoModelForCausalLM 

import customStylesListPA as styles
import modelsListPA as models
import scripts.PixArt_pipeline as pipeline
import scripts.controlnet_pixart as controlnet

class PixArtStorage:
    lastSeed = -1
    galleryIndex = 0
    lastPrompt = None
    lastNegative = None
    pos_embeds = None
    pos_attention = None
    neg_embeds = None
    neg_attention = None
    noiseRGBA = [0.0, 0.0, 0.0, 0.0]
    captionToPrompt = False
    sendAccessToken = False
    locked = False     #   for preventing changes to the following volatile state while generating
    karras = False
    vpred = False
    resolutionBin = True


from transformers import T5EncoderModel, T5Tokenizer
#from scripts.PixArt_pipeline import PixArtPipeline_DoE_combined

from diffusers import PixArtTransformer2DModel
from diffusers import AutoencoderKL
from diffusers import ConsistencyDecoderVAE
from diffusers import DEISMultistepScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, UniPCMultistepScheduler, DDPMScheduler
from diffusers import SASolverScheduler, LCMScheduler
from peft import PeftModel

from diffusers.utils.torch_utils import randn_tensor

# modules/infotext_utils.py
def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text

    return json.dumps(text, ensure_ascii=False)

# modules/processing.py
def create_infotext(model, positive_prompt, negative_prompt, guidance_scale, guidance_rescale, guidance_cutoff, steps, DMDstep, seed, scheduler, width, height):
    karras = " : Karras" if PixArtStorage.karras == True else ""
    vpred = " : V-Prediction" if PixArtStorage.vpred == True else ""
    isDMD = "PixArt-Alpha-DMD" in model
    generation_params = {
        "Size": f"{width}x{height}",
        "Seed": seed,
        "Scheduler": f"{scheduler}{karras}{vpred}",
        "Steps": steps if not isDMD else None,
        "CFG": f"{guidance_scale} ({guidance_rescale}) [{guidance_cutoff}]",
        "DMDstep": DMDstep if isDMD else None,
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None
    }
#add i2i marker?
    prompt_text = f"Prompt: {positive_prompt}\n"
    if negative_prompt != "":
        prompt_text += (f"Negative: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])
    noise_text = f"\nInitial noise: {PixArtStorage.noiseRGBA}" if PixArtStorage.noiseRGBA[3] != 0.0 else ""

    return f"Model: {model}\n{prompt_text}{generation_params_text}{noise_text}"

def predict(positive_prompt, negative_prompt, model, vae, width, height, guidance_scale, guidance_rescale, guidance_cutoff, num_steps, DMDstep, sampling_seed, num_images, scheduler, i2iSource, i2iDenoise, maskSource, maskCutOff, style, controlNetImage, controlNet, controlNetStrength, controlNetStart, controlNetEnd, *args):
    
    access_token = 0
    if PixArtStorage.sendAccessToken == True:
        try:
            with open('huggingface_access_token.txt', 'r') as file:
                access_token = file.read().strip()
        except:
            print ("PixArt: couldn't load 'huggingface_access_token.txt' from the webui directory. Will not be able to download/update gated models. Local cache will work.")

    torch.set_grad_enabled(False)

    ####    identify model type based on name
    isFlash = "flash-pixart" in model
    isSigma = "PixArt-Sigma" in model
    isDMD = "PixArt-Alpha-DMD" in model
    isLCM = "PixArt-LCM" in model

    if model in models.models_list_alpha:
        isSigma = False
    if model in models.models_list_sigma:
        isSigma = True

    useConsistencyVAE = (isSigma == 0) and (vae == 1)

    if not isSigma and controlNet != 0 and controlNetImage != None and controlNetStrength > 0.0:
        useControlNet = ['raulc0399/pixart-alpha-hed-controlnet'][controlNet-1]
    else:
        controlNetImage = None
        controlNetStrength = 0.0
        useControlNet = None


    if style != 0:
        positive_prompt = styles.styles_list[style][1].replace("{prompt}", positive_prompt)
        negative_prompt = styles.styles_list[style][2] + negative_prompt

    if i2iSource == None:
        maskSource = None
        i2iDenoise = 1
    if i2iDenoise < (num_steps + 1) / 1000:
        i2iDenoise = (num_steps + 1) / 1000

    if maskSource == None:
        maskCutOff = 1.0

    if PixArtStorage.resolutionBin == False:
        width  = (width  // 16) * 16
        height = (height // 16) * 16

    from diffusers.utils import logging
    logging.set_verbosity(logging.WARN)       #   download information is useful

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(sampling_seed)
    PixArtStorage.lastSeed = fixed_seed


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


    logging.set_verbosity(logging.ERROR)       #   avoid some console spam from Alpha models missing keys

    if isFlash:
        pipeModel = "PixArt-alpha/PixArt-XL-2-1024-MS"
    elif isSigma:
        pipeModel = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    else:
        pipeModel = model

    pipe = pipeline.PixArtPipeline_DoE_combined.from_pretrained(
        pipeModel,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=None,
        vae=None,
        controlnet=None,
        torch_dtype=torch.float16, )
    del tokenizer, text_encoder

    if isDMD:
        negative_prompt = ""

    if useCachedEmbeds == False:
        pos_embeds, pos_attention, neg_embeds, neg_attention = pipe.encode_prompt(positive_prompt, negative_prompt=negative_prompt)

        del pipe.tokenizer, pipe.text_encoder
        pipe.tokenizer = None
        pipe.text_encoder = None

        PixArtStorage.pos_embeds    = pos_embeds.to('cuda').to(torch.float16)
        PixArtStorage.neg_embeds    = neg_embeds.to('cuda').to(torch.float16)
        PixArtStorage.pos_attention = pos_attention.to('cuda').to(torch.float16)
        PixArtStorage.neg_attention = neg_attention.to('cuda').to(torch.float16)

        del pos_embeds, neg_embeds, pos_attention, neg_attention

        PixArtStorage.lastPrompt = positive_prompt
        PixArtStorage.lastNegative = negative_prompt

    gc.collect()
    torch.cuda.empty_cache()

    pipe.to('cuda')
####    load transformer, same process for Alpha and Sigma
    base = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS" if isSigma else "PixArt-alpha/PixArt-XL-2-1024-MS"
    if isFlash: #   base is an attempt at future proofing - base flash pipeline is currently always Alpha
        # Load LoRA
        pipe.transformer = PixArtTransformer2DModel.from_pretrained(
            base,
            subfolder="transformer",
            torch_dtype=torch.float16
        )
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            "jasperai/flash-pixart"
        )
    else:
        try:
            pipe.transformer = PixArtTransformer2DModel.from_pretrained(
                model,
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder='transformer',
                torch_dtype=torch.float16,
#                low_cpu_mem_usage=False,
#                device_map=None, 
                token=access_token,
            )
#            config=base)
        except:
            print ("PixArt: failed to load transformer. Repository may be gated and require a huggingface access token. See 'README.md'.")
            PixArtStorage.locked = False
            return gr.Button.update(value='Generate', variant='primary', interactive=True), None

##    # LoRA model -can't find examples in necessary form
    # loraLocation = ".//models//diffusers//PixArtLora//Wednesday.safetensors"
    # loraName = "Wednesday.safetensors"
    # pipe.transformer = PeftModel.from_pretrained(
        # pipe.transformer,
        # loraLocation,
        # adapter_name=loraName,
        # config=None,
        # local_files_only=True
    # )

####    VAEs are same for Alpha, and for Sigma. Sigma already shared, now Alpha is too.

    if useConsistencyVAE:   #   option for Alpha models
        pipe.vae = ConsistencyDecoderVAE.from_pretrained(
            "openai/consistency-decoder",
            local_files_only=False, cache_dir=".//models//diffusers//",
            torch_dtype=torch.float16)
    else:    
        cachedVAE = ".//models//diffusers//pixart_T5_fp16//vaeSigma" if isSigma else ".//models//diffusers//pixart_T5_fp16//vaeAlpha"
        sourceVAE = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers" if isSigma else "PixArt-alpha/PixArt-XL-2-1024-MS"

        try:
            pipe.vae = AutoencoderKL.from_pretrained(cachedVAE, local_files_only=True, variant="fp16", torch_dtype=torch.float16)
        except:
            pipe.vae = AutoencoderKL.from_pretrained(
                sourceVAE,
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder="vae",
                use_safetensors=True,
                torch_dtype=torch.float16, )

            ##  now save the converted fp16 VAE model to local cache, only needs done once
            pipe.vae.to(torch.float16)
            pipe.vae.save_pretrained(
                cachedVAE,
                safe_serialization=True,
                variant="fp16", )
            print ("Saved fp16 " + ("Sigma" if isSigma else "Alpha") + " VAE, will use this from now on.")

#    pipe.vae.enable_tiling(True) #make optional/based on dimensions

    if useControlNet:
        pipe.controlnet=controlnet.PixArtControlNetAdapterModel.from_pretrained(
            useControlNet,
            cache_dir=".//models//diffusers//",
            use_safetensors=True,
            torch_dtype=torch.float16
        )

    pipe.enable_model_cpu_offload()

    #   if using resolution_binning, must use adjusted width/height here (don't overwrite values)

    if PixArtStorage.resolutionBin:
        if pipe.transformer.config.sample_size == 256:
            aspect_ratio_bin = pipeline.ASPECT_RATIO_2048_BIN
        elif pipe.transformer.config.sample_size == 128:
            aspect_ratio_bin = pipeline.ASPECT_RATIO_1024_BIN
        elif pipe.transformer.config.sample_size == 64:
            aspect_ratio_bin = pipeline.ASPECT_RATIO_512_BIN
        elif pipe.transformer.config.sample_size == 32:
            aspect_ratio_bin = pipeline.ASPECT_RATIO_256_BIN
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

    #   always generate the noise here
    generator = [torch.Generator(device='cpu').manual_seed(fixed_seed+i) for i in range(num_images)]
    latents = randn_tensor(shape, generator=generator).to('cuda').to(torch.float16)
    #regen the generator to minimise differences between single/batch - might still be different - batch processing could use different pytorch kernels
    del generator
    generator = torch.Generator(device='cpu').manual_seed(14641)

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
    elif scheduler == 'LCM':
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif scheduler == "SA-solver":
        pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config, algorithm_type='data_prediction')
    elif scheduler == 'UniPC':
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#   else uses default set by model

    pipe.scheduler.config.num_train_timesteps = int(1000 * i2iDenoise)

    if hasattr(pipe.scheduler.config, 'use_karras_sigmas'):
        pipe.scheduler.config.use_karras_sigmas = PixArtStorage.karras
        
    if PixArtStorage.vpred == True:
        pipe.scheduler.config.prediction_type = 'v_prediction'

    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        output = pipe(
            generator                       = generator,
            latents                         = latents,                          #   initial noise, possibly with colour biasing

            image                           = i2iSource,
            mask_image                      = maskSource,
            strength                        = i2iDenoise,
            mask_cutoff                     = maskCutOff,
            control_image                   = controlNetImage,
            controlnet_conditioning_scale   = controlNetStrength,
            control_guidance_start          = controlNetStart,
            control_guidance_end            = controlNetEnd,

            num_inference_steps             = num_steps,
            num_images_per_prompt           = num_images,
            height                          = height,
            width                           = width,
            guidance_scale                  = guidance_scale,
            guidance_rescale                = guidance_rescale,
            guidance_cutoff                 = guidance_cutoff,
            prompt_embeds                   = PixArtStorage.pos_embeds,
            negative_prompt_embeds          = PixArtStorage.neg_embeds,
            prompt_attention_mask           = PixArtStorage.pos_attention,
            negative_prompt_attention_mask  = PixArtStorage.neg_attention,
            use_resolution_binning          = PixArtStorage.resolutionBin,
            timesteps                       = timesteps,
            output_type                     = "pil",
            isSigma                         = isSigma,
        ).images

#   vae uses lots of VRAM, especially with batches, maybe worth output to latent, free memory
#   but seem to need vae loaded earlier anyway

    del pipe.transformer, generator, latents

#    gc.collect()
#    torch.cuda.empty_cache()

    result = []
    for image in output:
        info=create_infotext(
            model,
            positive_prompt, negative_prompt,
            guidance_scale, guidance_rescale, guidance_cutoff,
            num_steps, DMDstep,
            fixed_seed, scheduler,
            width, height)

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

    PixArtStorage.locked = False
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
                                                         torch_dtype=torch.float16, 
                                                         cache_dir=".//models//diffusers//", 
                                                         trust_remote_code=True).to('cuda')
        processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', #-large
                                                  torch_dtype=torch.float32, 
                                                  cache_dir=".//models//diffusers//", 
                                                  trust_remote_code=True)

        result = ''
        prompts = ['<MORE_DETAILED_CAPTION>']

        for p in prompts:
            inputs = processor(text=p, images=image, return_tensors="pt")
            inputs.to('cuda').to(torch.float16)
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
            del parsed_answer
            if p != prompts[-1]:
                result += ' | \n'

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
    def toggleAccess ():
        PixArtStorage.sendAccessToken ^= True
        return gr.Button.update(variant=['secondary', 'primary'][PixArtStorage.sendAccessToken])

    #   these are volatile state, should not be changed during generation
    def toggleKarras ():
        if not PixArtStorage.locked:
            PixArtStorage.karras ^= True
        return gr.Button.update(variant='primary' if PixArtStorage.karras == True else 'secondary',
                                value='\U0001D40A' if PixArtStorage.karras == True else '\U0001D542')
    def toggleResBin ():
        if not PixArtStorage.locked:
            PixArtStorage.resolutionBin ^= True
        return gr.Button.update(variant='primary' if PixArtStorage.resolutionBin == True else 'secondary',
                                value='\U0001D401' if PixArtStorage.resolutionBin == True else '\U0001D539')
#    def toggleVP ():
#        if not PixArtStorage.locked:
#           PixArtStorage.vpred ^= True
#        return gr.Button.update(variant='primary' if PixArtStorage.vpred == True else 'secondary',
#                                value='\U0001D415' if PixArtStorage.vpred == True else '\U0001D54D')

    def toggleGenerate (R, G, B, A):
        PixArtStorage.noiseRGBA = [R, G, B, A]
        PixArtStorage.locked = True
        return gr.Button.update(value='...', variant='secondary', interactive=False)

    schedulerList = ["default", "DDPM", "DEIS", "DPM++ 2M", "DPM++ 2M SDE", "DPM", "DPM SDE",
                     "Euler", "Euler A", "LCM", "SA-solver", "UniPC", ]

    def parsePrompt (positive, negative, width, height, seed, scheduler, steps, cfg, guidance_rescale, guidance_cutoff, nr, ng, nb, ns):
        p = positive.split('\n')
        lineCount = len(p)

        negative = ''
        
        if "Prompt" != p[0] and "Prompt: " != p[0][0:8]:               #   civitAI style special case
            positive = p[0]
            l = 1
            while (l < lineCount) and not (p[l][0:17] == "Negative prompt: " or p[l][0:7] == "Steps: " or p[l][0:6] == "Size: "):
                if p[l] != '':
                    positive += '\n' + p[l]
                l += 1
        
        for l in range(lineCount):
            if "Prompt" == p[l][0:6]:
                if ": " == p[l][6:8]:                                   #   mine
                    positive = str(p[l][8:])
                    c = 1
                elif "Prompt" == p[l] and (l+1 < lineCount):            #   webUI
                    positive = p[l+1]
                    c = 2
                else:
                    continue

                while (l+c < lineCount) and not (p[l+c][0:10] == "Negative: " or p[l+c][0:15] == "Negative Prompt" or p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size: "):
                    if p[l+c] != '':
                        positive += '\n' + p[l+c]
                    c += 1
                l += 1

            elif "Negative" == p[l][0:8]:
                if ": " == p[l][8:10]:                                  #   mine
                    negative = str(p[l][10:])
                    c = 1
                elif " prompt: " == p[l][8:17]:                         #   civitAI
                    negative = str(p[l][17:])
                    c = 1
                elif " Prompt" == p[l][8:15] and (l+1 < lineCount):     #   webUI
                    negative = p[l+1]
                    c = 2
                else:
                    continue
                
                while (l+c < lineCount) and not (p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size: "):
                    if p[l+c] != '':
                        negative += '\n' + p[l+c]
                    c += 1
                l += 1

            elif "Initial noise: " == str(p[l][0:15]):
                noiseRGBA = str(p[l][16:-1]).split(',')
                nr = float(noiseRGBA[0])
                ng = float(noiseRGBA[1])
                nb = float(noiseRGBA[2])
                ns = float(noiseRGBA[3])
            else:
                params = p[l].split(',')
                for k in range(len(params)):
                    pairs = params[k].strip().split(' ')
                    match pairs[0]:
                        case "Size:":
                            size = pairs[1].split('x')
                            width = int(size[0])
                            height = int(size[1])
                        case "Seed:":
                            seed = int(pairs[1])
                        case "Sampler:":
                            sched = ' '.join(pairs[1:])
                            if sched in schedulerList:
                                scheduler = sched
                        case "Scheduler:":
                            sched = ' '.join(pairs[1:])
                            if sched in schedulerList:
                                scheduler = sched
                        case "Steps(Prior/Decoder):":
                            steps = str(pairs[1]).split('/')
                            steps = int(steps[0])
                        case "Steps:":
                            steps = int(pairs[1])
                        case "CFG":
                            if "scale:" == pairs[1]:
                                cfg = float(pairs[2])
                        case "CFG:":
                            cfg = float(pairs[1])
                            if len(pairs) == 4:
                                guidance_rescale = float(pairs[2].strip('\(\)'))
                                guidance_cutoff = float(pairs[3].strip('\[\]'))
                        case "width:":
                            width = float(pairs[1])
                        case "height:":
                            height = float(pairs[1])
        return positive, negative, width, height, seed, scheduler, steps, cfg, guidance_rescale, guidance_cutoff, nr, ng, nb, ns


    resolutionList256 = [
        (512, 128),     (432, 144),     (352, 176),     (320, 196),     (304, 208),
        (256, 256), 
        (208, 304),     (196, 320),     (176, 352),     (144, 432),     (128, 512)
    ]
    resolutionList512 = [
        (1024, 256),    (864, 288),     (704, 352),     (640, 384),     (608, 416),
        (512, 512), 
        (416, 608),     (384, 640),     (352, 704),     (288, 864),     (256, 1024)
    ]
    resolutionList1024 = [
        (2048, 512),    (1728, 576),    (1408, 704),    (1280, 768),    (1216, 832),
        (1024, 1024),
        (832, 1216),    (768, 1280),    (704, 1408),    (576, 1728),    (512, 2048)
    ]
    resolutionList2048 = [
        (4096, 1024),   (3456, 1152),   (2816, 1408),   (2560, 1536),   (2432, 1664),
        (2048, 2048),
        (1664, 2432),   (1536, 2560),   (1408, 2816),   (1152, 3456),   (1024, 4096)
    ]


    def updateWH (dims, w, h):
        #   returns None to dimensions dropdown so that it doesn't show as being set to particular values
        #   width/height could be manually changed, making that display inaccurate and preventing immediate reselection of that option
        #   passing by value because of odd gradio bug? when using index can either update displayed list correctly, or get values correctly, not both
        wh = dims.split('\u00D7')
        return None, int(wh[0]), int(wh[1])

    def processCN (image, method):
        if image:
            if method == 1:     # generate HED edge
                try:
                    from controlnet_aux import HEDdetector
                    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
                    hed_edge = hed(image)
                    return hed_edge
                except:
                    print ("Need controlAux package to preprocess.")
                    return image
        return image

    with gr.Blocks() as pixartsigma2_block:
        with ResizeHandleRow():
            with gr.Column():
                with gr.Row():
                    model = gr.Dropdown(models_list, label='Model', value=defaultModel, type='value', scale=2)
                    access = ToolButton(value='\U0001F917', variant='secondary')
                    vae = gr.Dropdown(["default", "consistency"], label='VAE', value='default', type='index', scale=1)
                    scheduler = gr.Dropdown(schedulerList,
                        label='Sampler', value="UniPC", type='value', scale=1)
                    karras = ToolButton(value="\U0001D542", variant='secondary', tooltip="use Karras sigmas")
#                    vpred = ToolButton(value="\U0001D54D", variant='secondary', tooltip="use v-prediction")

                with gr.Row():
                    positive_prompt = gr.Textbox(label='Prompt', placeholder='Enter a prompt here...', default='', lines=2)
                    parse = ToolButton(value="↙️", variant='secondary', tooltip="parse")

                with gr.Row():
                    negative_prompt = gr.Textbox(label='Negative', placeholder='', lines=2)
                    style = gr.Dropdown([x[0] for x in styles.styles_list], label='Style', value="(None)", type='index', scale=0)
                with gr.Row():
                    width = gr.Slider(label='Width', minimum=128, maximum=4096, step=16, value=defaultWidth, elem_id="PixArtSigma_width")
                    swapper = ToolButton(value="\U000021C4")
                    height = gr.Slider(label='Height', minimum=128, maximum=4096, step=16, value=defaultHeight, elem_id="PixArtSigma_height")
                    resBin = ToolButton(value="\U0001D401", variant='primary', tooltip="use resolution binning")
                    dims = gr.Dropdown([f'{i} \u00D7 {j}' for i,j in resolutionList1024],
                                        label='Quickset', type='value', scale=0)

                with gr.Row():
                    guidance_scale = gr.Slider(label='CFG', minimum=1, maximum=8, step=0.1, value=4.0, scale=1, visible=True)
                    guidance_rescale = gr.Slider(label='rescale CFG', minimum=0.00, maximum=1.0, step=0.01, value=0.0, precision=0.01, scale=1)
                    guidance_cutoff = gr.Slider(label='CFG cutoff after step', minimum=0.00, maximum=1.0, step=0.01, value=1.0, precision=0.01, scale=1)
                    DMDstep = gr.Slider(label='Timestep for DMD', minimum=1, maximum=999, step=1, value=400, scale=2, visible=False)
                with gr.Row():
                    steps = gr.Slider(label='Steps', minimum=1, maximum=60, step=1, value=20, scale=2, visible=True)
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

                with gr.Accordion(label='ControlNet (α only)', open=False):
                    with gr.Row():
                        CNSource = gr.Image(label='control image', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        with gr.Column():
                            CNMethod = gr.Dropdown(['(None)', 'HED edge'], label='method', value='(None)', type='index', multiselect=False, scale=1)
#                            CNProcess = gr.Button(value='Preprocess input image')
                            CNStrength = gr.Slider(label='Strength', minimum=0.00, maximum=1.0, step=0.01, value=0.8)
                            CNStart = gr.Slider(label='Start step', minimum=0.00, maximum=1.0, step=0.01, value=0.0)
                            CNEnd = gr.Slider(label='End step', minimum=0.00, maximum=1.0, step=0.01, value=0.8)

                with gr.Accordion(label='image to image', open=False):
                    with gr.Row():
                        i2iSource = gr.Image(label='image to image source', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        maskSource = gr.Image(label='source mask', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        with gr.Column():
                            i2iDenoise = gr.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                            i2iSetWH = gr.Button(value='Set Width / Height from image')
                            i2iFromGallery = gr.Button(value='Get image from gallery')
                            with gr.Row():
                                i2iCaption = gr.Button(value='Caption this image (Florence-2)', scale=7)
                                toPrompt = ToolButton(value='P', variant='secondary')
                            maskCut = gr.Slider(label='Ignore Mask after step', minimum=0.00, maximum=1.0, step=0.01, value=1.0)

                ctrls = [positive_prompt, negative_prompt, model, vae, width, height, guidance_scale, guidance_rescale, guidance_cutoff, steps, DMDstep, sampling_seed, batch_size, scheduler, i2iSource, i2iDenoise, maskSource, maskCut, style, CNSource, CNMethod, CNStrength, CNStart, CNEnd]
                
                parseCtrls = [positive_prompt, negative_prompt, width, height, sampling_seed, scheduler, steps, guidance_scale, guidance_rescale, guidance_cutoff, initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA]

            with gr.Column():
                generate_button = gr.Button(value="Generate", variant='primary', visible=True)
                output_gallery = gr.Gallery(label='Output', height="80vh",
                                            show_label=False, object_fit='contain', visible=True, columns=1, preview=True)
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


        def show_steps(model):
            if "PixArt-Alpha-DMD" in model:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(interactive=False), gr.update(visible=True)
            else:
                return gr.update(visible=True),  gr.update(visible=True),  gr.update(visible=True),  gr.update(interactive=True),  gr.update(visible=False)
        def set_dims(model):
            if "256" in model:
                resList = resolutionList256
            elif "512" in model:
                resList = resolutionList512
            elif "2K" in model or "2048" in model:
                resList = resolutionList2048
            else:
                resList = resolutionList1024

            choices = [f'{i} \u00D7 {j}' for i,j in resList]
            return gr.update(choices=choices)

        model.change(
            fn=show_steps,
            inputs=model,
            outputs=[guidance_scale, guidance_rescale, guidance_cutoff, steps, DMDstep],
            show_progress=False
        )
        model.change(
            fn=set_dims,
            inputs=model,
            outputs=dims,
            show_progress=False
        )
#        vpred.click(toggleVP, inputs=[], outputs=vpred)
#        CNProcess.click(processCN, inputs=[CNSource, CNMethod], outputs=[CNSource])
        dims.input(updateWH, inputs=[dims, width, height], outputs=[dims, width, height], show_progress=False)
        parse.click(parsePrompt, inputs=parseCtrls, outputs=parseCtrls, show_progress=False)
        access.click(toggleAccess, inputs=[], outputs=access)
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

        generate_button.click(predict, inputs=ctrls, outputs=[generate_button, output_gallery])
        generate_button.click(toggleGenerate, inputs=[initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA], outputs=[generate_button])

    return [(pixartsigma2_block, "PixArtSigma", "pixart_sigma")]

script_callbacks.on_ui_tabs(on_ui_tabs)

