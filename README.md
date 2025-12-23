# nnsight Image-to-Image Example

Simple example demonstrating how to use `nnsight` with Stable Diffusion's **Image-to-Image** pipeline to trace and intercept internal activations.

## Setup

This project uses `uv` for dependency management.

### 1. Install uv (if needed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

```bash
# Initialize uv project (if needed) and sync dependencies from pyproject.toml
uv sync
```

This will automatically create the virtual environment and install all required packages.

### 3. Activate Environment

```bash
source .venv/bin/activate
```

## Running the Example

Activate your environment and run the script:

```bash
source .venv/bin/activate
python main.py
```

This will:
1. Create a synthetic test input image (`input_image.png`)
2. Load a lightweight Stable Diffusion model (`segmind/tiny-sd`)
3. Run an **Image-to-Image** generation with prompt + image
4. Trace and capture the UNet's `conv_out` layer activations using nnsight
5. Save the final generated image to `img2img_output.png`

## Key Concept

To use Img2Img with nnsight, initialize the `DiffusionModel` with the specific pipeline class:

```python
from diffusers import StableDiffusionImg2ImgPipeline
from nnsight.modeling.diffusion import DiffusionModel

model = DiffusionModel(
    "model_id", 
    automodel=StableDiffusionImg2ImgPipeline,  # <--- Important!
)

# Pass both prompt and image
with model.generate(prompt, image=input_image) as tracer:
    # Trace internals...
```
