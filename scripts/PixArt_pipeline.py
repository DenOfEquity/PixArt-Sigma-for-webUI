####    THIS IS noUnload BRANCH, keeps models in RAM/VRAM to reduce loading time

# Copyright 2024 PixArt-Alpha Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
import inspect
import re
import urllib.parse as ul
from typing import Callable, List, Optional, Tuple, Union

import torch
from transformers import T5EncoderModel, T5TokenizerFast

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor, PixArtImageProcessor
from diffusers.models import AutoencoderKL, PixArtTransformer2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import (
    BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.pag.pag_utils import PAGMixin

from scripts.controlnet_pixart import PixArtControlNetAdapterModel, PixArtControlNetTransformerModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy


ASPECT_RATIO_2048_BIN = {
    "0.25": [1024.0, 4096.0],
    "0.26": [1024.0, 3968.0],
    "0.27": [1024.0, 3840.0],
    "0.28": [1024.0, 3712.0],
    "0.32": [1152.0, 3584.0],
    "0.33": [1152.0, 3456.0],
    "0.35": [1152.0, 3328.0],
    "0.4": [1280.0, 3200.0],
    "0.42": [1280.0, 3072.0],
    "0.48": [1408.0, 2944.0],
    "0.5": [1408.0, 2816.0],
    "0.52": [1408.0, 2688.0],
    "0.57": [1536.0, 2688.0],
    "0.6": [1536.0, 2560.0],
    "0.68": [1664.0, 2432.0],
    "0.72": [1664.0, 2304.0],
    "0.78": [1792.0, 2304.0],
    "0.82": [1792.0, 2176.0],
    "0.88": [1920.0, 2176.0],
    "0.94": [1920.0, 2048.0],
    "1.0": [2048.0, 2048.0],
    "1.07": [2048.0, 1920.0],
    "1.13": [2176.0, 1920.0],
    "1.21": [2176.0, 1792.0],
    "1.29": [2304.0, 1792.0],
    "1.38": [2304.0, 1664.0],
    "1.46": [2432.0, 1664.0],
    "1.67": [2560.0, 1536.0],
    "1.75": [2688.0, 1536.0],
    "2.0": [2816.0, 1408.0],
    "2.09": [2944.0, 1408.0],
    "2.4": [3072.0, 1280.0],
    "2.5": [3200.0, 1280.0],
    "2.89": [3328.0, 1152.0],
    "3.0": [3456.0, 1152.0],
    "3.11": [3584.0, 1152.0],
    "3.62": [3712.0, 1024.0],
    "3.75": [3840.0, 1024.0],
    "3.88": [3968.0, 1024.0],
    "4.0": [4096.0, 1024.0],
}

ASPECT_RATIO_1024_BIN = {
    "0.25": [512.0, 2048.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_512_BIN = {
    "0.25": [256.0, 1024.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "3.0": [864.0, 288.0],
    "4.0": [1024.0, 256.0],
}

ASPECT_RATIO_256_BIN = {
    "0.25": [128.0, 512.0],
    "0.28": [128.0, 464.0],
    "0.32": [144.0, 448.0],
    "0.33": [144.0, 432.0],
    "0.35": [144.0, 416.0],
    "0.4": [160.0, 400.0],
    "0.42": [160.0, 384.0],
    "0.48": [176.0, 368.0],
    "0.5": [176.0, 352.0],
    "0.52": [176.0, 336.0],
    "0.57": [192.0, 336.0],
    "0.6": [192.0, 320.0],
    "0.68": [208.0, 304.0],
    "0.72": [208.0, 288.0],
    "0.78": [224.0, 288.0],
    "0.82": [224.0, 272.0],
    "0.88": [240.0, 272.0],
    "0.94": [240.0, 256.0],
    "1.0": [256.0, 256.0],
    "1.07": [256.0, 240.0],
    "1.13": [272.0, 240.0],
    "1.21": [272.0, 224.0],
    "1.29": [288.0, 224.0],
    "1.38": [288.0, 208.0],
    "1.46": [304.0, 208.0],
    "1.67": [320.0, 192.0],
    "1.75": [336.0, 192.0],
    "2.0": [352.0, 176.0],
    "2.09": [368.0, 176.0],
    "2.4": [384.0, 160.0],
    "2.5": [400.0, 160.0],
    "3.0": [432.0, 144.0],
    "4.0": [512.0, 128.0],
}


# Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,                                              # (`SchedulerMixin`): scheduler to get timesteps from.
    num_inference_steps: Optional[int] = None,              # (`int`):            number of diffusion steps used  - priority 3
    device: Optional[Union[str, torch.device]] = None,      # (`str` or `torch.device`, *optional*): device to move timesteps to. If `None`, not moved.
    timesteps: Optional[List[int]] = None,                  # (`List[int]`, *optional*): custom timesteps, length overrides num_inference_steps - priority 1
    sigmas: Optional[List[float]] = None,                   # (`List[float]`, *optional*): custom sigmas, length overrides num_inference_steps - priority 2
    **kwargs,
):
    #   stop aborting on recoverable errors!
    #   default to using timesteps
    if timesteps is not None and "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None and "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

#from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin #maybe PAGMixin

class PixArtPipeline_DoE_combined(DiffusionPipeline, PAGMixin):

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    _optional_components = ["tokenizer", "text_encoder"]
    model_cpu_offload_seq = "text_encoder->transformer->refiner->controlnet->vae"

    def __init__(
        self,
        tokenizer: T5TokenizerFast,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: PixArtTransformer2DModel,
        scheduler: DPMSolverMultistepScheduler,
        refiner: Optional[PixArtTransformer2DModel] = None,
        controlnet: Optional[PixArtControlNetAdapterModel] = None,
        pag_applied_layers: Union[str, List[str]] = ["blocks.14"],  # 1st transformer block
    ):
        super().__init__()
        
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler, controlnet=controlnet, refiner=refiner
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        self.latent_channels = (
            self.vae.config.latent_channels
            if hasattr(self, "vae") and self.vae is not None
            else 4
        )
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor  = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, vae_latent_channels=self.latent_channels, 
                                                 do_resize=False, do_normalize=False, do_binarize=False, do_convert_grayscale=True)
        self.set_pag_applied_layers(pag_applied_layers)

    # Modified from diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt
    def encode_prompt(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        device: Optional[torch.device] = None,
        isSigma: bool = True,
        **kwargs,
    ):

        if device is None:
            device = self._execution_device

        max_length = 300 if isSigma else 120
        
        def prompt_and_weights (tokenizer, prompt):
            prompt = self._text_preprocessing(prompt)
            promptSplit = prompt[0].split(' ')
            cleanedPrompt = ' '.join((t.split(':')[0] for t in promptSplit))
            weights = []

            for t in promptSplit:
                t = t.split(':')
                if len(t) == 1:
                    weight = 1.0
                elif t[1] == '':
                    weight = 1.0
                else:
                    weight = float(t[1])

                text_inputsX = tokenizer(
                    t[0],
                    padding=False,
                    max_length=max_length,
                    truncation=True,
                    return_attention_mask=False,
                    add_special_tokens=False,
                    return_tensors="pt",
                )

                tokenLength = len(text_inputsX.input_ids[0])
                for w in range(tokenLength):
                    weights.append(weight)
            
            return cleanedPrompt, weights

        positive_prompt, positive_weights = prompt_and_weights(self.tokenizer, prompt)
        negative_prompt, negative_weights = prompt_and_weights(self.tokenizer, negative_prompt)

        text_inputs = self.tokenizer(
            [positive_prompt] + [negative_prompt],
            padding=True,
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        positive_attention = text_inputs.attention_mask[0:1]
        negative_attention = text_inputs.attention_mask[1:]

        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(device), attention_mask=text_inputs.attention_mask.to(device))

        max_tokens = min (len(positive_weights), len(prompt_embeds[0][0]))
        positive_mean_before = prompt_embeds[0][0].mean()
        for p in range(max_tokens):
            prompt_embeds[0][0][p] *= positive_weights[p]
        positive_mean_after = prompt_embeds[0][0].mean()
        prompt_embeds[0][0] *= positive_mean_before / positive_mean_after
            
        max_tokens = min (len(negative_weights), len(prompt_embeds[0][1]))      #   should be same
        negative_mean_before = prompt_embeds[0][1].mean()
        for p in range(max_tokens):
            prompt_embeds[0][1][p] *= negative_weights[p]
        negative_mean_after = prompt_embeds[0][1].mean()
        prompt_embeds[0][1] *= negative_mean_before / negative_mean_after

        positive_prompt_embeds = prompt_embeds[0][0].unsqueeze(0)
        negative_prompt_embeds = prompt_embeds[0][1].unsqueeze(0)

        # prompt = self._text_preprocessing(prompt)
        # text_inputs = self.tokenizer(
            # prompt,
            # padding="max_length",
            # max_length=max_length,
            # truncation=True,
            # add_special_tokens=True,
            # return_tensors="pt",
        # )
        # text_input_ids = text_inputs.input_ids
        # positive_attention = text_inputs.attention_mask
        # positive_attention = positive_attention.to(device)

        # prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=positive_attention)
        # positive_prompt_embeds = prompt_embeds[0]

        # prompt = self._text_preprocessing(negative_prompt)
        # text_inputs = self.tokenizer(
            # prompt,
            # padding="max_length",
            # max_length=max_length,
            # truncation=True,
            # add_special_tokens=True,
            # return_tensors="pt",
        # )
        # text_input_ids = text_inputs.input_ids
        # negative_attention = text_inputs.attention_mask
        # negative_attention = negative_attention.to(device)

        # prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=negative_attention)
        # negative_prompt_embeds = prompt_embeds[0]

        return positive_prompt_embeds, positive_attention, negative_prompt_embeds, negative_attention

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs


    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start


    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing
    def _text_preprocessing(self, text):
        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            text = text.lower().strip()
            return text

        return [process(t) for t in text]


    @torch.no_grad()
    def __call__(
        self,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.5,
        guidance_rescale: float = 0.0,
        guidance_cutoff: float = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        use_resolution_binning: bool = True,
        isSigma: bool = True,
        isDMD: bool = False,
        
        image: PipelineImageInput = None,
        strength: float = 0.6,
        mask_image: PipelineImageInput = None,
        mask_cutoff: float = 1.0,
        control_image: PipelineImageInput = None,
        controlnet_conditioning_scale: float = 1.0,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,

        pag_scale: float = 3.0,
        pag_adaptive_scale: float = 0.0,

        noUnload: Optional[bool] = False,

        **kwargs,
    ) -> torch.Tensor:

        doDiffDiff = True if (image and mask_image) else False
        self._pag_scale = pag_scale
        self._pag_adaptive_scale = pag_adaptive_scale

        # 0.01 repeat prompt embeds to match num_images_per_prompt
        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
        negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)

        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        if use_resolution_binning:
            if self.transformer.config.sample_size == 256:
                aspect_ratio_bin = ASPECT_RATIO_2048_BIN
            if self.transformer.config.sample_size == 128:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_256_BIN
            else:
                raise ValueError("Invalid sample size")

            height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)


        # 2. Default height and width to transformer

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt - already done
        if self.do_perturbed_attention_guidance:
            prompt_embeds = self._prepare_perturbed_attention_guidance(
                prompt_embeds, negative_prompt_embeds, do_classifier_free_guidance
            )
            prompt_attention_mask = self._prepare_perturbed_attention_guidance(
                prompt_attention_mask, negative_prompt_attention_mask, do_classifier_free_guidance
            )
        elif do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        components_multipler = 1
        if self.do_perturbed_attention_guidance:
            components_multipler += 1
        if do_classifier_free_guidance:
            components_multipler += 1

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, sigmas)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels

        if image is not None:
            noise = latents

            # 4. Prepare timesteps
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

            # 3. Preprocess image
            image = self.image_processor.preprocess(image, height=height, width=width).to('cuda').to(torch.float16)
            image_latents = self.vae.encode(image).latent_dist.sample(generator)
            image_latents *= self.vae.config.scaling_factor * self.scheduler.init_noise_sigma
            image_latents = image_latents.repeat(num_images_per_prompt, 1, 1, 1)

            # add noise to image latents
            if strength < 1.0:
                ts = torch.tensor([timesteps[0]], device='cuda')
                ts = ts[:1].repeat(num_images_per_prompt)
                latents = self.scheduler.add_noise(image_latents, noise, ts)

            if mask_image is not None:
                # 5.1. Prepare masked latent variables
                mask = self.mask_processor.preprocess(mask_image.resize((width//8, height//8))).to(device='cuda', dtype=torch.float16)

        if self.controlnet != None and control_image is not None:
            control_image = self.image_processor.preprocess(control_image.resize((width, height))).to(device='cuda', dtype=torch.float16)
            control_latents = self.vae.encode(control_image).latent_dist.sample() * self.vae.config.scaling_factor
            del control_image
            control_latents = control_latents.repeat(num_images_per_prompt, 1, 1, 1)
            if components_multipler > 1:
                control_latents = torch.cat([control_latents] * components_multipler)

            # change to the controlnet transformer model
            self.transformer = PixArtControlNetTransformerModel(
                transformer=self.transformer,
                controlnet=self.controlnet,
            ).to('cuda')
        else:
            control_latents = None

        if self.do_perturbed_attention_guidance:
            original_attn_proc = self.transformer.attn_processors
            self._set_pag_attn_processor(
                pag_applied_layers=self.pag_applied_layers,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )


        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if not isSigma and self.transformer.config.sample_size == 128:
            resolution = torch.tensor([height, width]).repeat(num_images_per_prompt, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(num_images_per_prompt, 1)
            resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
            aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)

            if components_multipler > 1:
                resolution = torch.cat([resolution] * components_multipler, dim=0)
                aspect_ratio = torch.cat([aspect_ratio] * components_multipler, dim=0)

            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if doDiffDiff and float((i+1) / num_timesteps) <= mask_cutoff:
                    tmask = (mask >= float((i+1) / num_timesteps))
                    ts = torch.tensor([t], device='cuda')
                    ts = ts[:1].repeat(num_images_per_prompt)
                    init_latents_proper = self.scheduler.add_noise(image_latents, noise, ts)
                    latents = (init_latents_proper * ~tmask) + (latents * tmask)

                if float((i+1) / num_timesteps) > guidance_cutoff and guidance_scale != 1.0 and PAG_scale == 0.0:
                    do_classifier_free_guidance = False
                    guidance_scale = 1.0
                    components_multipler -= 1
                    
                    prompt_embeds = prompt_embeds[components_multipler * num_images_per_prompt:]
                    prompt_attention_mask = prompt_attention_mask[components_multipler * num_images_per_prompt:]
                    if not isSigma and self.transformer.config.sample_size == 128:
                        resolution = resolution[components_multipler * num_images_per_prompt:]
                        aspect_ratio = aspect_ratio[components_multipler * num_images_per_prompt:]
                        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}
                    if control_latents is not None:
                        control_latents = control_latents[components_multipler * num_images_per_prompt:]


                latent_model_input = torch.cat([latents] * components_multipler, dim=0)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(latent_model_input.shape[0])

                # predict noise model_output
                if float((i+1) / num_timesteps) < control_guidance_start:
                    control_cond = None
                elif float((i+1) / num_timesteps) > control_guidance_end:
                    control_cond = None
                else:
                    control_cond = control_latents
                 
                if control_cond is not None:
                    noise_pred = self.transformer(
                        latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        timestep=current_timestep,
                        controlnet_cond=control_cond,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    if self.refiner != None and t <= 400:
                        noise_pred = self.refiner(
                            latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_attention_mask,
                            timestep=current_timestep,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
                    else:
                        noise_pred = self.transformer(
                            latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_attention_mask,
                            timestep=current_timestep,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
#   if noUnload: manually move transformer /refiner between cpu/gpu ?
#   don't need the VRAM?

                #   perform guidance
                if self.do_perturbed_attention_guidance:
                    noise_pred = self._apply_perturbed_attention_guidance(
                        noise_pred, do_classifier_free_guidance, guidance_scale, t
                    )
                elif do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # compute previous image: x_t -> x_t-1
                if isDMD and num_inference_steps == 1:
                    # For DMD one step sampling: https://arxiv.org/abs/2311.18828
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).pred_original_sample
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if doDiffDiff and 1.0 <= mask_cutoff:
            tmask = (mask >= 1.0)
            latents = (image_latents * ~tmask) + (latents * tmask)

        if control_latents is not None:
            del control_latents
            self.transformer = self.transformer.transformer     #   undo controlnet change

        if self.do_perturbed_attention_guidance:
            self.transformer.set_attn_processor(original_attn_proc)

        # Offload all models
        self.maybe_free_model_hooks()

        return latents
