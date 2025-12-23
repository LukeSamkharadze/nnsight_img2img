"""
nnsight example demonstrating image-to-image pipeline with step-by-step activation capturing.
Captures text encoder output and UNet activations at specific time steps.
"""

import torch
from PIL import Image
from nnsight.modeling.diffusion import DiffusionModel
from diffusers import StableDiffusionImg2ImgPipeline


def create_test_image(size=(512, 512)):
    """Create a simple test image (gradient)."""
    import numpy as np
    x = np.linspace(0, 255, size[0]).astype(np.uint8)
    y = np.linspace(0, 255, size[1]).astype(np.uint8)
    xx, yy = np.meshgrid(x, y)
    img_array = np.stack([xx, yy, (xx + yy) // 2], axis=-1).astype(np.uint8)
    return Image.fromarray(img_array, mode='RGB')


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Create test input image
    input_image = create_test_image()
    
    # Load the img2img pipeline
    print("Loading Stable Diffusion Img2Img model...")
    model = DiffusionModel(
        "segmind/tiny-sd",
        automodel=StableDiffusionImg2ImgPipeline,
        torch_dtype=torch.float16,
        dispatch=True
    ).to(device)
    
    prompt = "A mountain landscape"
    
    print(f"Prompt: '{prompt}'")
    print("Running img2img with tracing to capture activations at multiple steps...\n")
    
    # Use generate to trace
    with model.generate(
        prompt,
        image=input_image,
        strength=0.75,
        num_inference_steps=5, # Small number of steps for testing
        seed=42,
    ) as tracer:
        
        # 1. Capture Text Encoder Output (happens once at start)
        # Note: In tiny-sd/SD pipelines, text_encoder output is usually last_hidden_state
        print("Tracing text encoder...")
        text_emb = model.text_encoder.output.last_hidden_state.save()
        
        # 2. Capture UNet outputs at specific steps
        # Step 0 (First denoising step)
        print("Tracing UNet step 0...")
        unet_step0 = model.unet.conv_out.output.save()
        
        # Step 1 (Second denoising step) - access via .next() within the generation loop context
        model.unet.next() # Advance to next call
        print("Tracing UNet step 1...")
        unet_step1 = model.unet.conv_out.output.save()
        
        # Step 2
        model.unet.next()
        print("Tracing UNet step 2...")
        unet_step2 = model.unet.conv_out.output.save()

    
    # Print results
    print("\n=== nnsight Multi-Step Capture Results ===")
    
    print(f"\nText Encoder Output Shape: {text_emb.shape}")
    print(f"Text Encoder values (subset):\n{text_emb[0, :3, :5]}")
    
    print(f"\nUNet Step 0 Output Shape: {unet_step0.shape}")
    print(f"UNet Step 0 val: {unet_step0[0, 0, 0, 0].item():.4f}")
    
    print(f"\nUNet Step 1 Output Shape: {unet_step1.shape}")
    print(f"UNet Step 1 val: {unet_step1[0, 0, 0, 0].item():.4f}")
    
    print(f"\nUNet Step 2 Output Shape: {unet_step2.shape}")
    print(f"UNet Step 2 val: {unet_step2[0, 0, 0, 0].item():.4f}")
    
    print("\nVerification successful: Captured activations for text encoder and multiple UNet steps!")


if __name__ == "__main__":
    main()
