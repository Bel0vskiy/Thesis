![header](https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,100:2d4a7a&height=200&text=Evaluating%20The%20Faithfulness%20of%20Medical%20VLMs&fontSize=25&fontColor=e0e8f0&animation=fadeIn)

## Notebooks

Gemma-3 and MedGemma share a single notebook (`deletion_medgemma.ipynb`) because both are loaded identically through the standard `transformers` `AutoModelForCausalLM` path thus only the model id differs. However, LLaVA 1.5 and LLaVA-Med, by contrast, require separate notebooks. LLaVA-Med can only be loaded through the official Microsoft LLaVA-Med repository (`llava.model.builder.load_pretrained_model`), which ships its own model builder, tokeniser utilities, and image-token constants, hence two different notebooks exist for llava and llavamed.