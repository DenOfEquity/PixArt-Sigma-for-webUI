import gradio as gr
import torch
import gc
import json
from transformers import T5EncoderModel

from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.rng import create_generator
from modules.shared import opts
from modules.ui_components import ResizeHandleRow
import modules.infotext_utils as parameters_copypaste

torch.backends.cuda.enable_mem_efficient_sdp(True)


lastSeed = -1
galleryIndex = 0


from diffusers import PixArtSigmaPipeline, PixArtAlphaPipeline, Transformer2DModel
from diffusers import DEISMultistepScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from diffusers import EulerAncestralDiscreteScheduler, UniPCMultistepScheduler


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

#   should save actual model
    prompt_text = f"Prompt: {positive_prompt}\n"
    if negative_prompt != "":
        prompt_text += (f"Negative: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])

    return f"Model: {model}\n{prompt_text}{generation_params_text}"

def predict(positive_prompt, negative_prompt, model, width, height, guidance_scale, num_steps, sampling_seed, num_images, scheduler, *args):

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(sampling_seed)
    global lastSeed
    lastSeed = fixed_seed

    isSigma = "PixArt-Sigma" in model

#    algorithm_type = args.algorithm
#    beta_schedule = args.beta_schedule
#    use_lu_lambdas = args.use_lu_lambdas

    pipe = None

#   the T5 model is always the same

    text_encoder = None
##  first, try to load converted fp16 T5 model
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
##        text_encoder.to(torch.float16)
##        text_encoder.save_pretrained(
##            ".//models//diffusers//pixart_T5_fp16",
##            variant="fp16",
##            safe_serialization=True, )


    if isSigma:
        pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
            local_files_only=False, cache_dir=".//models//diffusers//",
            text_encoder=text_encoder,
            transformer=None,
            torch_dtype=torch.float16, )
    else:
        pipe = PixArtAlphaPipeline.from_pretrained(
            model,
            text_encoder=text_encoder,
            transformer=None,
            torch_dtype=torch.float16, )        

    pipe.enable_attention_slicing("max")    #win at start of loading, costs at end?, reduces gap to fp16 version

    with torch.no_grad():
        pos_embeds, pos_attention, neg_embeds, neg_attention = pipe.encode_prompt(positive_prompt, negative_prompt=negative_prompt)

    pipe.text_encoder = None
    del text_encoder

    gc.collect()
    torch.cuda.empty_cache()

    if isSigma:
        pipe.transformer = Transformer2DModel.from_pretrained(
            model,
            local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder='transformer',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
            device_map=None, )
    else:   #   sigma path also works for alpha, but this way suppresses a warning message in console
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        pipe = PixArtAlphaPipeline.from_pretrained(
            model,
            text_encoder=None,
            torch_dtype=torch.float16,
        ).to("cuda")


##        #test, save fp16 transformer, not so important
##        pipe.transformer.save_pretrained(
##            save_directory=".//models//diffusers//",    #save here, default name (doesn't indicate original)
##            variant="fp16",
##            safe_serialization=True, )


#   if not Windows:    
#       pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)

    pipe.to('cuda')
    pipe.enable_model_cpu_offload()
    
    generator = [torch.Generator().manual_seed(fixed_seed+i) for i in range(num_images)]

    pos_embeds    = pos_embeds.to('cuda').to(torch.float16)
    neg_embeds    = neg_embeds.to('cuda').to(torch.float16)
    pos_attention = pos_attention.to('cuda').to(torch.float16)
    neg_attention = neg_attention.to('cuda').to(torch.float16)

    #   information!
#    print (pipe.scheduler.compatibles)
## <class 'diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler'>,
## <class 'diffusers.schedulers.scheduling_dpmsolver_sde.DPMSolverSDEScheduler'>,
## <class 'diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler'>,
## <class 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler'>,
## <class 'diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler'>,
## <class 'diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler'>
##




    if scheduler == 'DEIS':
        pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DPM':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DPMsinglestep':
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Euler':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Euler A':
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'UniPC':
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


#   else uses default set by model

##    pipe.scheduler.beta_schedule  = beta_schedule
##    pipe.scheduler.algorithm_type = algorithm_type
##    pipe.scheduler.use_karras_sigmas = karras
##    pipe.scheduler.use_lu_lambdas = use_lu_lambdas

    output = pipe(
        negative_prompt=None, 
        num_inference_steps=num_steps,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        prompt_embeds=pos_embeds,
        negative_prompt_embeds=neg_embeds,
        prompt_attention_mask=pos_attention,
        negative_prompt_attention_mask=neg_attention,
        num_images_per_prompt=num_images,
        output_type="pil",  #"latent"
        generator=generator,
    ).images

    del pipe

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

    return result



def on_ui_tabs():
    models_list_alpha = ["PixArt-alpha/PixArt-XL-2-256x256",
                         "PixArt-alpha/PixArt-XL-2-512x512",
                         "PixArt-alpha/PixArt-XL-2-1024-MS"]

    models_list_sigma = ["PixArt-alpha/PixArt-Sigma-XL-2-256x256",
                         "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
                         "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                         "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS"]
#    custom_models_alpha = ["artificialguybr/Fascinatio-PixartAlpha1024-Finetuned"]
#    custom_models_sigma = ["frutiemax/VintageKnockers-Pixart-Sigma-XL-2-512-MS",
#                           "frutiemax/VintageKnockers-Pixart-Sigma-XL-2-1024-MS"
#                           ]

    models_list = models_list_alpha + models_list_sigma
    
    from modules.ui_components import ToolButton

    def getGalleryIndex (evt: gr.SelectData):
        global galleryIndex
        galleryIndex = evt.index

    def reuseLastSeed ():
        global lastSeed, galleryIndex
        return lastSeed + galleryIndex
        
    def randomSeed ():
        return -1

    with gr.Blocks() as pixartsigma_block:
        with ResizeHandleRow():
            with gr.Column():
                positive_prompt = gr.Textbox(label='Prompt', placeholder='Enter a prompt here...', default='')
                negative_prompt = gr.Textbox(label='Negative', placeholder='')
                with gr.Row():
                    model = gr.Dropdown(models_list,
                                        label='Model', value="PixArt-alpha/PixArt-Sigma-XL-2-512-MS", type='value', scale=2)
                    sampling_seed = gr.Number(label='Seed', value=-1, precision=0, scale=1)
                    random = ToolButton(value="\U0001f3b2\ufe0f")
                    reuseSeed = ToolButton(value="\u267b\ufe0f")
                with gr.Row():
                    scheduler = gr.Dropdown(["default",
                                             "DEIS",
                                             "DPM",
                                             "DPMsinglestep",
                                             "Euler",
                                             "Euler A",
                                             "UniPC"],
                        label='Sampler', value="default", type='value', scale=1)
                    guidance_scale = gr.Slider(label='CFG', minimum=1, maximum=8, step=0.5, value=4.0, scale=2)
                    steps = gr.Slider(label='Steps', minimum=1, maximum=60, step=1, value=20, scale=2)
                    batch_size = gr.Number(label='Batch Size', minimum=1, maximum=9, step=1, value=1, precision=0, scale=0)
                with gr.Row():
                    width = gr.Slider(label='Width', minimum=128, maximum=4096, step=8, value=512, elem_id="PixArtSigma_width")
                    swapper = ToolButton(value="\U000021C5")
                    height = gr.Slider(label='Height', minimum=128, maximum=4096, step=8, value=768, elem_id="PixArtSigma_height")


                ctrls = [positive_prompt, negative_prompt, model, width, height, guidance_scale, steps, sampling_seed, batch_size, scheduler]

            with gr.Column():
                generate_button = gr.Button(value="Generate")
                output_gallery = gr.Gallery(label='Output', height=shared.opts.gallery_height or None,
                                            show_label=False, object_fit='contain', visible=True, columns=3, preview=True)
#   gallery movement buttons don't work, others do
#   caption not displaying linebreaks

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
