"""
nnsight example demonstrating image-to-image pipeline with DiffusionModel.
This tests whether we can input images to the DiffusionModel for img2img generation.
"""

import torch
from PIL import Image
from nnsight.modeling.diffusion import DiffusionModel
from diffusers import StableDiffusionImg2ImgPipeline


def create_test_image(size=(512, 512)):
    """Create a simple test image (gradient)."""
    import numpy as np
    
    # Create a simple gradient image
    x = np.linspace(0, 255, size[0]).astype(np.uint8)
    y = np.linspace(0, 255, size[1]).astype(np.uint8)
    xx, yy = np.meshgrid(x, y)
    
    # RGB gradient
    img_array = np.stack([xx, yy, (xx + yy) // 2], axis=-1).astype(np.uint8)
    return Image.fromarray(img_array, mode='RGB')


def main():
    # Check for available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Create a test input image
    print("Creating test input image...")
    input_image = create_test_image(size=(512, 512))
    input_image.save("input_image.png")
    print("Saved input image to: input_image.png")
    
    # Load the img2img pipeline with nnsight wrapper
    print("\nLoading Stable Diffusion Img2Img model...")
    model = DiffusionModel(
        "segmind/tiny-sd",  # Using tiny-sd for faster testing
        automodel=StableDiffusionImg2ImgPipeline,
        torch_dtype=torch.float16,
        dispatch=True
    ).to(device)
    
    prompt = "A beautiful landscape painting with mountains and sunset"
    
    print(f"\nPrompt: '{prompt}'")
    print("Running img2img with tracing to capture UNet activations...\n")
    
    # Use generate with tracing - pass image as keyword argument
    with model.generate(
        prompt,
        image=input_image,
        strength=0.75,
        num_inference_steps=20,
        seed=42,
    ) as tracer:
        # Capture activations from the UNet
        unet_output = model.unet.conv_out.output.save()
    
    # Print results
    print("=== nnsight Img2Img Example ===\n")
    print(f"Input image size: {input_image.size}")
    print(f"Prompt: '{prompt}'")
    print(f"\nUNet conv_out output shape: {unet_output.shape}")
    print(f"UNet conv_out output (sample):\n{unet_output[0, :, :3, :3]}")
    
    # Generate actual image without tracing
    print("\nGenerating final image...")
    result = model.generate(
        prompt,
        image=input_image,
        strength=0.75,
        num_inference_steps=20,
        seed=42,
        trace=False
    )
    
    # Save the output
    output_path = "img2img_output.png"
    result.images[0].save(output_path)
    print(f"Output image saved to: {output_path}")
    print("\nSuccess! nnsight supports img2img pipelines!")


if __name__ == "__main__":
    main()
