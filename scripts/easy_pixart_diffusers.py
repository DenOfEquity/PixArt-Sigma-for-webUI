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


from diffusers import PixArtSigmaPipeline, Transformer2DModel, DEISMultistepScheduler, DPMSolverMultistepScheduler
import argparse
import pathlib
from pathlib import Path
import sys

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))


# modules/infotext_utils.py
def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text

    return json.dumps(text, ensure_ascii=False)

# modules/processing.py
def create_infotext(positive_prompt, negative_prompt, guidance_scale, steps, seed, width, height):
    generation_params = {
        "Model": "PixArtSigma",
        "Size": f"{width}x{height}",
        "Seed": seed,
        "Steps": steps,
        "CFG": guidance_scale,
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None
    }

    generation_params_text = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])

    prompt_text = positive_prompt
    negative_prompt_text = f"\nNegative prompt: {negative_prompt}" if negative_prompt else ""

    return f"{prompt_text}{negative_prompt_text}\n{generation_params_text}".strip()

def predict(positive_prompt, negative_prompt, model, width, height, guidance_scale, num_steps, sampling_seed, num_images, *args):

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(sampling_seed)
    global lastSeed
    lastSeed = fixed_seed

    scheduler_type = 'DPM'#'deis'#'DPM'#args.scheduler
    karras = True#args.karras

#    algorithm_type = args.algorithm
#    beta_schedule = args.beta_schedule
#    use_lu_lambdas = args.use_lu_lambdas

    pipe = None

    text_encoder = None
##  first, try to load converted fp16 T5 model
    try:
        text_encoder = T5EncoderModel.from_pretrained(
            ".//models//diffusers//pixart_sigma_sdxlvae_T5_diffusers_fp16",
            local_files_only=True, cache_dir=".//models//diffusers//",
            variant="fp16",
            torch_dtype=torch.float16,
            device_map="auto",
        )
    except:
        ##  fetch the T5 model, ~20gigs
        ##  convert to f16 for use
        ##  specifying cache directory because transformers will cache to a different location than diffusers, so you can have 20gigs of T5 twice
        text_encoder = T5EncoderModel.from_pretrained(
            "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
            local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder="text_encoder",
            use_safetensors=True,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        ##  now save the converted fp16 T5 model to local cache, only needs done once
#        text_encoder.to(torch.float16)
#        text_encoder.save_pretrained(f"pixart_sigma_sdxlvae_T5_diffusers_fp16", cache_dir=".//models//diffusers//", variant="fp16", safe_serialization=True)

    
    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        local_files_only=False, cache_dir=".//models//diffusers//",
        text_encoder=text_encoder,
        transformer=None,
        torch_dtype=torch.float16,
    )

    with torch.no_grad():
        prompt = positive_prompt
        negative = negative_prompt
        prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompt, negative_prompt=negative)


    pipe.text_encoder = None
    del text_encoder

    gc.collect()
    torch.cuda.empty_cache()

    pipe.transformer = Transformer2DModel.from_pretrained(model,
                                                          local_files_only=False, cache_dir=".//models//diffusers//",
                                                          subfolder='transformer',
                                                          torch_dtype=torch.float16)
    pipe.to('cuda')
    pipe.enable_model_cpu_offload()
    
    generator = torch.Generator()
    generator = generator.manual_seed(fixed_seed)

    prompt_embeds = prompt_embeds.to('cuda').to(torch.float16)
    negative_embeds = negative_embeds.to('cuda').to(torch.float16)
    prompt_attention_mask = prompt_attention_mask.to('cuda').to(torch.float16)
    negative_prompt_attention_mask = negative_prompt_attention_mask.to('cuda').to(torch.float16)

    if scheduler_type == 'deis':
        pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

##    pipe.scheduler.beta_schedule  = beta_schedule
##    pipe.scheduler.algorithm_type = algorithm_type
##    pipe.scheduler.use_karras_sigmas = karras
##    pipe.scheduler.use_lu_lambdas = use_lu_lambdas

    output = pipe(
        negative_prompt=None, 
        num_inference_steps=num_steps,
        height=height,
        width=width,
        prompt_embeds=prompt_embeds,
        guidance_scale=guidance_scale,
        negative_prompt_embeds=negative_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        num_images_per_prompt=num_images,
        output_type="pil",  #"latent"
        generator=generator,
    ).images


##    with torch.no_grad():
##        images = pipe.vae.decode(output / pipe.vae.config.scaling_factor, return_dict=False)[0]
##        images = pipe.image_processor.postprocess(images, output_type="pil")

    for image in output:
        images.save_image(
            image,
            opts.outdir_samples or opts.outdir_txt2img_samples,
            "",
            fixed_seed,
            prompt,
            opts.samples_format,
            info=create_infotext(prompt, negative_prompt, guidance_scale, num_steps, fixed_seed, width, height)
        )

    return output



def on_ui_tabs():
    from modules.ui_components import ToolButton

    def reuseLastSeed ():
        global lastSeed
        return lastSeed
        
    def randomSeed ():
        return -1

    with gr.Blocks() as pixartsigma_block:
        with ResizeHandleRow():
            with gr.Column():
                prompt = gr.Textbox(label='Prompt', placeholder='Enter a prompt here...', default='')
                negative_prompt = gr.Textbox(label='Negative Prompt', placeholder='')
                with gr.Row():
                    model = gr.Dropdown(["PixArt-alpha/PixArt-Sigma-XL-2-256x256",
                                         "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
                                         "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                                         "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS"],
                                        label='Model', value="PixArt-alpha/PixArt-Sigma-XL-2-512-MS", type='value', scale=1)
                    sampling_seed = gr.Number(label='Seed', value=-1, precision=0, scale=1)
                    random = ToolButton(value="\U0001f3b2\ufe0f")
                    reuseSeed = ToolButton(value="\u267b\ufe0f")
                with gr.Row():
                    guidance_scale = gr.Slider(label='CFG', minimum=1, maximum=8, step=0.5, value=4.0, scale=2)
                    steps = gr.Slider(label='Steps', minimum=1, maximum=60, step=1, value=20, scale=2)
                    batch_size = gr.Slider(label='Batch Size', minimum=1, maximum=9, step=1, value=1)
                with gr.Row():
                    width = gr.Slider(label='Width', minimum=128, maximum=4096, step=8, value=512, elem_id="PixArtSigma_width")
                    swapper = ToolButton(value="\U000021C5")
                    height = gr.Slider(label='Height', minimum=128, maximum=4096, step=8, value=768, elem_id="PixArtSigma_height")


                ctrls = [prompt, negative_prompt, model, width, height, guidance_scale, steps, sampling_seed, batch_size]

            with gr.Column():
                generate_button = gr.Button(value="Generate")
                output_gallery = gr.Gallery(label='Output', height=shared.opts.gallery_height or None, show_label=False, object_fit='contain', visible=True, columns=3, preview=True)
                
                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname, source_text_component=prompt, source_image_component=output_gallery,
                    ))

        swapper.click(fn=None, _js="function(){switchWidthHeight('PixArtSigma')}", inputs=None, outputs=None, show_progress=False)
        random.click(randomSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        reuseSeed.click(reuseLastSeed, inputs=[], outputs=sampling_seed, show_progress=False)

        generate_button.click(predict, inputs=ctrls, outputs=[output_gallery])
    return [(pixartsigma_block, "PixArtSigma", "pixart_sigma")]

script_callbacks.on_ui_tabs(on_ui_tabs)
