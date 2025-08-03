import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image, ImageEnhance

try:
    # Load a model optimized for realistic human details
    model_id = "dreamlike-art/dreamlike-photoreal-2.0"  # Better for photorealistic humans; fallback: "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # FP16 for 4GB GPU
        use_auth_token=False
    ).to("cuda")

    # Use EulerDiscreteScheduler for sharper details
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # Apply memory-saving techniques
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()  # Uncomment if xformers installed: pip install xformers

    # Detailed prompt to improve hand and anatomy rendering
    prompt = (
        "A photorealistic depiction of a man with a helmet, wearing a leather racing suit, "
        "firmly gripping the handlebars of a sleek Kawasaki Ninja bike on a race track, "
        "sunny day, sharp details, realistic anatomy, vibrant colors"
    )
    negative_prompt = (
        "blurry, low-resolution, unrealistic, distorted, low-quality, pixelated, oversaturated, "
        "artifacts, deformed hands, extra fingers, missing fingers, malformed limbs, unnatural anatomy"
    )

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Generate image with optimized settings
    generator = torch.Generator("cuda").manual_seed(42)  # Fixed seed for reproducibility
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=75,  # Increased for better detail
        guidance_scale=9.5,      # Stronger prompt adherence
        height=512, width=512,   # Kept at 512x512 for 4GB GPU (try 640x640 if stable)
        generator=generator
    ).images[0]

    # Save the generated image
    image.save("optimized_image.png")

    # Post-process to enhance sharpness and contrast
    img = Image.open("optimized_image.png")
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    sharp_img = enhancer.enhance(2.0)  # Adjust factor as needed
    # Enhance contrast
    contrast_enhancer = ImageEnhance.Contrast(sharp_img)
    final_img = contrast_enhancer.enhance(1.2)  # Slight contrast boost
    final_img.save("enhanced_image.png")
    print("✅ Enhanced image saved successfully!")

    # Display the enhanced image
    plt.imshow(final_img)
    plt.axis('off')  # Hide axis for better viewing
    plt.show()

except Exception as e:
    print(f"❌ Error occurred: {str(e)}")
    print("Try reinstalling dependencies: pip install --upgrade torch torchvision transformers diffusers")
    print("Ensure no conflicting local files named 'torch.py' or 'diffusers.py'.")
    print("If memory issues occur, reduce num_inference_steps to 50 or resolution to height=448, width=448.")
