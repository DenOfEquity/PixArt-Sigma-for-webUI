## PixArt Sigma (and Alpha) for webui ##
### Forge tested, probably A1111 too ###
I don't think there is anything Forge specific here.
### works for me (tm) on 8Gb VRAM, 16Gb RAM (GTX1070) ###

---
### downloads models on demand - minimum will be ~20Gb ###
### needs updated *diffusers* ###
be sure you are updating the right one, in Forge:
```
*forge directory*\system\python\Lib\site-packages
```
```
pip install --upgrade -t .\ git+https://github.com/huggingface/diffusers
```
You should end up with v0.28.0 (as of 30/04/2024).
There is code for earlier versions of diffusers, but it didn't work with the 2K model and needed updating for 256 and 1024 (at least for me).

### Alpha needs updated *transformers* and *tokenizers* ###
Easiest way to achieve this is to edit **requirements.text** and **requirements_versions.txt** in the webUI folder. Otherwise starting the webUI will undo the tokenizers upgrade.
```
tokenizers>=0.19
transformers==4.40
```
These upgrades didn't break anything for me.

---
At your own risk. This is barely tested, and even then only on my computer.
Models will be downloaded automatically, on demand (so if you never generate with the 256 model, it'll never be downloaded). The T5 text encoder is around 18Gb and the image models are about 2.3Gb each.
I preferentially load a fp16 version of the T5 model. Fall back is to the full model which is converted to fp16 when used. Line to save it locally (in models/diffusers) is commented out. If you'd like to do as I do, uncomment the cast line and *text_encoder.save_pretrained* line (from line 99). This significantly improves the speed of the first processing step, from ~45 seconds to ~8 seconds.

I can generate using all models, though the 2K model does hit shared memory a lot, so is significantly slower.

---
### added 01/05/2024 ###
* samplers
* captions in gallery (where linebreaks?)
* correct seeds for batches

### added 02/05/2024 ###
* support for PixArt-Alpha models - they use the same T5 text encoder.
* fixed (but still commented out) saving of fp16 text encoder. Previously might not have saved in the right place.

### to do ###
models still have duplicate VAEs, maybe neatly consolidate them too

add LCM models?

---
prompt: portrait photograph, woman with red hair, wearing green blazer over yellow tshirt and blue trousers, on sunny beach with dark clouds on horizon

![portrait photograph, woman with red hair, wearing green blazer over yellow tshirt and blue trousers, on sunny beach with dark clouds on horizon](example.png "20 steps with 1024 model")

---

---
Thanks to:
	[frutiemax92](https://github.com/frutiemax92) for inference_pipeline.py
	[benjamin-bertram](https://github.com/benjamin-bertram/sdweb-easy-stablecascade-diffusers) for ui details