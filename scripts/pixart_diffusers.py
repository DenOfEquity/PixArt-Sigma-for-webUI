import gradio as gr
import math
import torch
import gc
import json

from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.rng import create_generator
from modules.shared import opts
from modules.ui_components import ResizeHandleRow
import modules.infotext_utils as parameters_copypaste

torch.backends.cuda.enable_mem_efficient_sdp(True)

class PixArtStorage:
    lastSeed = -1
    galleryIndex = 0
    lastPrompt = None
    lastNegative = None
    pos_embeds = None
    pos_attention = None
    neg_embeds = None
    neg_attention = None

from transformers import T5EncoderModel, T5Tokenizer
from diffusers import PixArtSigmaPipeline, PixArtAlphaPipeline, Transformer2DModel
from diffusers import AutoencoderKL
from diffusers import ConsistencyDecoderVAE
from diffusers import DEISMultistepScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers import SASolverScheduler


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
def create_infotext(model, positive_prompt, negative_prompt, guidance_scale, steps, seed, scheduler, width, height):
    generation_params = {
        "Size": f"{width}x{height}",
        "Seed": seed,
        "Scheduler": scheduler,
        "Steps": steps,
        "CFG": guidance_scale,
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None
    }

    prompt_text = f"Prompt: {positive_prompt}\n"
    if negative_prompt != "":
        prompt_text += (f"Negative: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])

    return f"Model: {model}\n{prompt_text}{generation_params_text}"

def predict(positive_prompt, negative_prompt, model, width, height, guidance_scale, num_steps, sampling_seed, num_images, scheduler, style, *args):

####    shamelessly copied from PixArt repo on Github
    styles_list = [
        ("", ""),
        ("cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
         "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"),
        ("cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
         "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"),
        ("anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
         "photo, deformed, black and white, realism, disfigured, low contrast"),
        ("manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
         "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style"),
        ("concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
         "photo, photorealistic, realism, ugly"),
        ("pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
         "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic"),
        ("ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
         "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white"),
        ("neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
         "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"),
        ("professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
         "ugly, deformed, noisy, low poly, blurry, painting"), ]


    if style != 0:
        p, n = styles_list[style]
        positive_prompt = p.replace("{prompt}", positive_prompt)
        negative_prompt = n + negative_prompt



    from diffusers.utils import logging
    logging.set_verbosity(logging.WARN)       #   download information is useful

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(sampling_seed)
    PixArtStorage.lastSeed = fixed_seed

    ####    identify model type basde on name
    isSigma = "PixArt-Sigma" in model
    isDMD = "PixArt-Alpha-DMD" in model
    isLCM = "PixArt-LCM" in model

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
                local_files_only=True,
                variant="fp16",
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



####    VAEs are consistent for Alpha, and for Sigma. Sigma already shared, now Alpha is too.

########    DMD uses consistencyVAE
    if isDMD:
        vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16)
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

    logging.set_verbosity(logging.ERROR)       #   avoid some console spam from Alpha models missing keys

    if isSigma:
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

#    pipe.enable_attention_slicing("max")

    if isDMD:
        negative_prompt = ""
        guidance_scale = 1
        num_steps = 1
    if isDMD or isLCM:
        scheduler = 'default'

    if useCachedEmbeds == False:
        with torch.no_grad():
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


####    load transformer, same for Alpha and Sigma
    pipe.transformer = Transformer2DModel.from_pretrained(
        model,
        local_files_only=False, cache_dir=".//models//diffusers//",
        subfolder='transformer',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        device_map=None, )

    pipe.to('cuda')
    pipe.enable_model_cpu_offload()




##        #test, save fp16 transformer, not so important
##        pipe.transformer.save_pretrained(
##            save_directory=".//models//diffusers//",    #save here, default name (doesn't indicate original)
##            variant="fp16",
##            safe_serialization=True, )


#   if not Windows:    
#       pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)

    
    generator = [torch.Generator().manual_seed(fixed_seed+i) for i in range(num_images)]



    if scheduler == 'DEIS':
        pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DPM':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DPMsinglestep':
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

##    pipe.scheduler.beta_schedule  = beta_schedule
##    pipe.scheduler.algorithm_type = algorithm_type
##    pipe.scheduler.use_karras_sigmas = karras
##    pipe.scheduler.use_lu_lambdas = use_lu_lambdas


####    generate some good? timesteps, this is an exponential-cosine blend, by me
##    CEB_timesteps = []
##    if num_steps != 1:
##        timestep_max = 1000
##        timestep_min = 5
##        K = (timestep_min / timestep_max)**(1/(num_steps-1))
##        E = timestep_max
##
##        for x in range(num_steps):
##            p = x / (num_steps-1)
##            C = timestep_min + 0.5*(timestep_max-timestep_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
##            CEB_timesteps.append(int(0.5 + C + p * (E - C)))
##            E *= K
##
##        print (CEB_timesteps)



    output = pipe(
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
        timesteps=[400] if isDMD else None,
    ).images

    del pipe.transformer, vae, generator
    pipe.transformer = None


    result = []
    
    for image in output:
        info=create_infotext(
            model,
            positive_prompt, negative_prompt,
            guidance_scale, num_steps,
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

    return result





def on_ui_tabs():
    styles_list = ["(None)",
                   "Cinematic", "Photographic",
                   "Anime", "Manga",
                   "Digital art", "Pixel art",
                   "Fantasy art", "Neonpunk", "3D model"
                  ]
   
    models_list_alpha = ["PixArt-alpha/PixArt-XL-2-256x256",
                         "PixArt-alpha/PixArt-XL-2-512x512",
                         "PixArt-alpha/PixArt-XL-2-1024-MS",
                         "PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512"]

    models_list_sigma = ["PixArt-alpha/PixArt-Sigma-XL-2-256x256",
                         "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
                         "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                         "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS",
                         "PixArt-alpha/PixArt-LCM-XL-2-1024-MS"]
#    custom_models_alpha = ["artificialguybr/Fascinatio-PixartAlpha1024-Finetuned"]
#    custom_models_sigma = ["frutiemax/VintageKnockers-Pixart-Sigma-XL-2-512-MS",
#                           "frutiemax/VintageKnockers-Pixart-Sigma-XL-2-1024-MS"
#                           ]

    models_list = models_list_alpha + models_list_sigma
    
    from modules.ui_components import ToolButton

    def getGalleryIndex (evt: gr.SelectData):
        PixArtStorage.galleryIndex = evt.index

    def reuseLastSeed ():
        return PixArtStorage.lastSeed + PixArtStorage.galleryIndex
        
    def randomSeed ():
        return -1

    with gr.Blocks() as pixartsigma_block:
        with ResizeHandleRow():
            with gr.Column():
                positive_prompt = gr.Textbox(label='Prompt', placeholder='Enter a prompt here...', default='')
                with gr.Row():
                    negative_prompt = gr.Textbox(label='Negative', placeholder='')
                    style = gr.Dropdown(styles_list, label='Style', value="(None)", type='index', scale=0)
                with gr.Row():
                    model = gr.Dropdown(models_list, label='Model', value="PixArt-alpha/PixArt-Sigma-XL-2-512-MS", type='value', scale=2)
                    sampling_seed = gr.Number(label='Seed', value=-1, precision=0, scale=1)
                    random = ToolButton(value="\U0001f3b2\ufe0f")
                    reuseSeed = ToolButton(value="\u267b\ufe0f")
                with gr.Row():
                    scheduler = gr.Dropdown(["default",
                                             "DEIS",
                                             "DPM",
                                             "DPM SDE",
                                             "DPMsinglestep",
                                             "Euler",
                                             "Euler A",
                                             "SA-solver",
                                             "UniPC",
                                             ],
                        label='Sampler', value="UniPC", type='value', scale=1)
                    guidance_scale = gr.Slider(label='CFG', minimum=1, maximum=8, step=0.5, value=4.0, scale=2)
                    steps = gr.Slider(label='Steps', minimum=1, maximum=60, step=1, value=20, scale=2)
                    batch_size = gr.Number(label='Batch Size', minimum=1, maximum=9, step=1, value=1, precision=0, scale=0)
                with gr.Row():
                    width = gr.Slider(label='Width', minimum=128, maximum=4096, step=8, value=512, elem_id="PixArtSigma_width")
                    swapper = ToolButton(value="\U000021C5")
                    height = gr.Slider(label='Height', minimum=128, maximum=4096, step=8, value=768, elem_id="PixArtSigma_height")


                ctrls = [positive_prompt, negative_prompt, model, width, height, guidance_scale, steps, sampling_seed, batch_size, scheduler, style]

            with gr.Column():
                generate_button = gr.Button(value="Generate")
                output_gallery = gr.Gallery(label='Output', height=shared.opts.gallery_height or None,
                                            show_label=False, object_fit='contain', visible=True, columns=3, preview=True)
#   gallery movement buttons don't work, others do
#   caption not displaying linebreaks, alt text does

                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname, source_text_component=positive_prompt, source_image_component=output_gallery,
                    ))

        swapper.click(fn=None, _js="function(){switchWidthHeight('PixArtSigma')}", inputs=None, outputs=None, show_progress=False)
        random.click(randomSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        reuseSeed.click(reuseLastSeed, inputs=[], outputs=sampling_seed, show_progress=False)


        output_gallery.select (fn=getGalleryIndex, inputs=[], outputs=[])

        generate_button.click(predict, inputs=ctrls, outputs=[output_gallery])
    return [(pixartsigma_block, "PixArtSigma", "pixart_sigma")]

script_callbacks.on_ui_tabs(on_ui_tabs)
