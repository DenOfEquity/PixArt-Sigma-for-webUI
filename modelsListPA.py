models_list_alpha = [
    "PixArt-alpha/PixArt-XL-2-256x256",                        #   smol
    "PixArt-alpha/PixArt-XL-2-512x512",
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    "jasperai/flash-pixart",
    "PixArt-alpha/PixArt-LCM-XL-2-1024-MS",                    #   flash-pixart better
    "PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512",              #   flash-pixart better
#    "artificialguybr/Fascinatio-PixartAlpha1024-Finetuned",
]
models_list_sigma = [
    "PixArt-alpha/PixArt-Sigma-XL-2-256x256",                  #   smol
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS",                    #   undertrained
    "ptx0/pixart-sigma",
#    "ptx0/pixart-reality-mix",                                 #   not working with standard pipelines, here for future, v-pred?
#    "frutiemax/VintageKnockers-Pixart-Sigma-XL-2-512-MS",      #   nsfw
#    "frutiemax/VintageKnockers-Pixart-Sigma-XL-2-1024-MS",     #   nsfw
]

defaultModel = models_list_sigma[2]
defaultWidth = 1024
defaultHeight = 1024

