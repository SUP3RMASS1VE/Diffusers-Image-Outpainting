# Import warning suppression module first
import fix_warnings

import gradio as gr
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download
import warnings
import os

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from PIL import Image, ImageDraw
import numpy as np

# Disable the warning about fp16 and non-fp16 filenames 
os.environ["DIFFUSERS_NO_SAFETENSORS_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=".*mixture of fp16 and non-fp16 filenames.*")

config_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="config_promax.json",
)

config = ControlNetModel_Union.load_config(config_file)
controlnet_model = ControlNetModel_Union.from_config(config)
model_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="diffusion_pytorch_model_promax.safetensors",
)
state_dict = load_state_dict(model_file)

# Load the controlnet model
model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
    controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
)
model.to(device="cuda", dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to("cuda")

# Load the pipeline
pipe = StableDiffusionXLFillPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0_Lightning",
    torch_dtype=torch.float16,
    vae=vae,
    controlnet=model,
    variant="fp16",
).to("cuda")

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)


def can_expand(source_width, source_height, target_width, target_height, alignment):
    """Checks if the image can be expanded based on the alignment."""
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True

def prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    target_size = (width, height)

    # Calculate the scaling factor to fit the image within the target size
    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    # Resize the source image to fit within target size
    source = image.resize((new_width, new_height), Image.LANCZOS)

    # Apply resize option using percentages
    if resize_option == "Full":
        resize_percentage = 100
    elif resize_option == "50%":
        resize_percentage = 50
    elif resize_option == "33%":
        resize_percentage = 33
    elif resize_option == "25%":
        resize_percentage = 25
    else:  # Custom
        resize_percentage = custom_resize_percentage

    # Calculate new dimensions based on percentage
    resize_factor = resize_percentage / 100
    new_width = int(source.width * resize_factor)
    new_height = int(source.height * resize_factor)

    # Ensure minimum size of 64 pixels
    new_width = max(new_width, 64)
    new_height = max(new_height, 64)

    # Resize the image
    source = source.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the overlap in pixels based on the percentage
    overlap_x = int(new_width * (overlap_percentage / 100))
    overlap_y = int(new_height * (overlap_percentage / 100))

    # Ensure minimum overlap of 1 pixel
    overlap_x = max(overlap_x, 1)
    overlap_y = max(overlap_y, 1)

    # Calculate margins based on alignment
    if alignment == "Middle":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - new_width
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = target_size[1] - new_height

    # Adjust margins to eliminate gaps
    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    # Create a new background image and paste the resized source image
    background = Image.new('RGB', target_size, (30, 30, 30))
    background.paste(source, (margin_x, margin_y))

    # Create the mask
    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    # Calculate overlap areas
    white_gaps_patch = 2

    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
    
    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height


    # Draw the mask
    mask_draw.rectangle([
        (left_overlap, top_overlap),
        (right_overlap, bottom_overlap)
    ], fill=0)

    return background, mask

def preview_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)
    
    # Create a preview image showing the mask
    preview = background.copy().convert('RGBA')
    
    # Create a semi-transparent green overlay (better visibility against dark background)
    overlay = Image.new('RGBA', background.size, (0, 255, 128, 100))  # Increased alpha for better visibility
    
    # Convert black pixels in the mask to semi-transparent overlay
    overlay_mask = Image.new('RGBA', background.size, (0, 0, 0, 0))
    overlay_mask.paste(overlay, (0, 0), mask)
    
    # Overlay the mask on the background
    preview = Image.alpha_composite(preview, overlay_mask)
    
    return preview

def infer(image, width, height, overlap_percentage, num_inference_steps, resize_option, custom_resize_percentage, prompt_input, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    background, mask = prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom)
    
    if not can_expand(background.width, background.height, width, height, alignment):
        alignment = "Middle"

    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)

    final_prompt = f"{prompt_input} , high quality, 4k"

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(final_prompt, "cuda", True)

    # Modified to only yield the final result
    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
        num_inference_steps=num_inference_steps
    ):
        pass

    image = image.convert("RGBA")
    cnet_image.paste(image, (0, 0), mask)

    # Return only the final result
    return cnet_image

def clear_result():
    """Clears the result Image."""
    return gr.update(value=None)

def preload_presets(target_ratio, ui_width, ui_height):
    """Updates the width and height sliders based on the selected aspect ratio."""
    if target_ratio == "9:16":
        changed_width = 720
        changed_height = 1280
        return changed_width, changed_height, gr.update()
    elif target_ratio == "16:9":
        changed_width = 1280
        changed_height = 720
        return changed_width, changed_height, gr.update()
    elif target_ratio == "1:1":
        changed_width = 1024
        changed_height = 1024
        return changed_width, changed_height, gr.update()
    elif target_ratio == "Custom":
        return ui_width, ui_height, gr.update(open=True)

def select_the_right_preset(user_width, user_height):
    if user_width == 720 and user_height == 1280:
        return "9:16"
    elif user_width == 1280 and user_height == 720:
        return "16:9"
    elif user_width == 1024 and user_height == 1024:
        return "1:1"
    else:
        return "Custom"

def toggle_custom_resize_slider(resize_option):
    return gr.update(visible=(resize_option == "Custom"))

def update_history(new_image, history):
    """Updates the history gallery with the new image."""
    if history is None:
        history = []
    history.insert(0, new_image)
    return history

css = """
.container {
    /* max-width: 1200px !important; */
    /* margin: 0 auto !important; */
}

.header {
    background: linear-gradient(90deg, #1a237e, #311b92);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}

.header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
}

.header p {
    margin: 8px 0 0;
    opacity: 0.9;
    font-size: 1.1rem;
}

.footer {
    background: #212121;
    padding: 15px;
    border-radius: 10px;
    margin-top: 20px;
    text-align: center;
    font-size: 0.9rem;
    color: #bbbbbb;
}

.footer a {
    color: #90caf9;
    text-decoration: none;
}

.tool-section {
    border: 1px solid #424242;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    background: #1e1e1e;
}

.tool-section-title {
    font-weight: 600;
    margin-bottom: 10px;
    color: #90caf9;
}

.controls-section {
    background: #2d2d2d;
    padding: 10px;
    border-radius: 8px;
}

.generate-button {
    background: linear-gradient(90deg, #3949ab, #5e35b1) !important;
    color: white !important;
}

.preview-button {
    background: #388e3c !important; 
    color: white !important;
}

.help-text {
    font-size: 0.85rem;
    color: #bdbdbd;
    margin-top: 5px;
    font-style: italic;
}

/* Tab styling */
.tab-nav {
    border-bottom: 2px solid #424242;
}

.tab-nav button {
    border-radius: 0 !important;
    font-weight: 600;
}

.tab-nav button.selected {
    border-bottom: 3px solid #90caf9 !important;
    color: #90caf9 !important;
}

/* Image containers */
.image-display {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.history-section {
    margin-top: 15px;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Monochrome(primary_hue="indigo", neutral_hue="slate", font=["Inter", "sans-serif"])) as demo:
    with gr.Column(elem_classes="container"):
        # Header section
        with gr.Group(elem_classes="header"):
            gr.HTML("""
                <h1>Diffusers Image Outpainting</h1>
                <p>Easily expand your image boundaries with AI-powered outpainting</p>
            """)

        # Main content section
        with gr.Tabs():
            with gr.TabItem("Create", elem_classes="tab-item"):
                with gr.Row():
                    # Left column - Input section
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="tool-section"):
                            gr.Markdown("### Input Image", elem_classes="tool-section-title")
                            input_image = gr.Image(
                                type="pil",
                                label="Upload or drag an image here",
                                elem_classes="image-display"
                            )
                            gr.Markdown("*For best results, use images with clear boundaries and content*", elem_classes="help-text")

                        with gr.Group(elem_classes="tool-section"):
                            gr.Markdown("### Generation Settings", elem_classes="tool-section-title")
                            with gr.Row():
                                with gr.Column(scale=2):
                                    prompt_input = gr.Textbox(
                                        label="Prompt", 
                                        placeholder="Describe what should appear in the expanded areas..."
                                    )
                                    gr.Markdown("*Adding descriptive details helps create better outpainting results*", elem_classes="help-text")
                                
                                with gr.Column(scale=1):
                                    run_button = gr.Button("Generate Outpainting", elem_classes="generate-button")

                            with gr.Row():
                                with gr.Column():
                                    target_ratio = gr.Radio(
                                        label="Canvas Size",
                                        choices=["9:16", "16:9", "1:1", "Custom"],
                                        value="9:16",
                                        scale=2
                                    )
                                
                                with gr.Column():  
                                    alignment_dropdown = gr.Dropdown(
                                        choices=["Middle", "Left", "Right", "Top", "Bottom"],
                                        value="Middle",
                                        label="Image Alignment"
                                    )
                                    gr.Markdown("*Choose where to position your original image*", elem_classes="help-text")

                        with gr.Accordion(label="Advanced Settings", open=False) as settings_panel:
                            with gr.Group(elem_classes="controls-section"):
                                with gr.Row():
                                    with gr.Column():
                                        width_slider = gr.Slider(
                                            label="Canvas Width",
                                            minimum=720,
                                            maximum=1536,
                                            step=8,
                                            value=720,
                                        )
                                    with gr.Column():
                                        height_slider = gr.Slider(
                                            label="Canvas Height",
                                            minimum=720,
                                            maximum=1536,
                                            step=8,
                                            value=1280,
                                        )
                                
                                num_inference_steps = gr.Slider(
                                    label="Generation Steps", 
                                    minimum=4, 
                                    maximum=12, 
                                    step=1, 
                                    value=8
                                )
                                gr.Markdown("*Higher values may produce better results but take longer*", elem_classes="help-text")
                                
                                with gr.Group():
                                    gr.Markdown("### Mask Settings")
                                    overlap_percentage = gr.Slider(
                                        label="Mask Overlap (%)",
                                        minimum=1,
                                        maximum=50,
                                        value=10,
                                        step=1
                                    )
                                    gr.Markdown("*Controls how much original image to blend with new content*", elem_classes="help-text")
                                    
                                    with gr.Row():
                                        with gr.Column():
                                            overlap_top = gr.Checkbox(label="Overlap Top", value=True)
                                            overlap_left = gr.Checkbox(label="Overlap Left", value=True)
                                        with gr.Column():
                                            overlap_right = gr.Checkbox(label="Overlap Right", value=True)
                                            overlap_bottom = gr.Checkbox(label="Overlap Bottom", value=True)
                                
                                with gr.Group():
                                    gr.Markdown("### Image Resize")
                                    with gr.Row():
                                        resize_option = gr.Radio(
                                            label="Resize input image",
                                            choices=["Full", "50%", "33%", "25%", "Custom"],
                                            value="Full"
                                        )
                                        custom_resize_percentage = gr.Slider(
                                            label="Custom resize (%)",
                                            minimum=1,
                                            maximum=100,
                                            step=1,
                                            value=50,
                                            visible=False
                                        )
                            
                            with gr.Row():
                                preview_button = gr.Button("Preview Mask & Alignment", elem_classes="preview-button")
                    
                    # Right column - Output section
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="tool-section"):
                            gr.Markdown("### Generated Result", elem_classes="tool-section-title")
                            result = gr.Image(
                                label="Outpainted Image",
                                elem_classes="image-display",
                                interactive=False,
                                show_download_button=True
                            )
                            use_as_input_button = gr.Button("Use as Input Image", visible=False)
                                
                        with gr.Group(elem_classes="tool-section"):
                            gr.Markdown("### Mask Preview", elem_classes="tool-section-title")
                            preview_image = gr.Image(
                                label="Mask & Alignment Preview",
                                elem_classes="image-display"
                            )
                            gr.Markdown("*Green areas show where new content will be generated*", elem_classes="help-text")
                
                with gr.Group(elem_classes="history-section"):
                    gr.Markdown("### Generation History", elem_classes="tool-section-title")
                    history_gallery = gr.Gallery(
                        label="Previous Results", 
                        columns=6, 
                        object_fit="contain", 
                        interactive=False,
                        elem_classes="image-display",
                        show_download_button=True
                    )
            
            with gr.TabItem("Help & Tips", elem_classes="tab-item"):
                gr.Markdown("""
                    ## How to Use
                    
                    1. **Upload an image** - Start by uploading the image you want to expand
                    2. **Choose canvas size** - Select a preset aspect ratio or customize dimensions
                    3. **Set alignment** - Position your image on the canvas
                    4. **Add a prompt** - Describe what should appear in the expanded areas
                    5. **Generate** - Click the Generate button to create your outpainted image
                    
                    ## Tips for Better Results
                    
                    - **Use clear images** with well-defined boundaries
                    - **Be specific in your prompts** to guide the outpainting
                    - **Experiment with mask overlap** to improve blending
                    - **Try different alignments** to get the desired expansion
                    - **Use the preview** to check mask positioning before generating
                    
                    ## About Outpainting
                    
                    Outpainting extends the boundaries of an existing image using AI to create new, 
                    contextually relevant content beyond the original borders. This tool uses Stable Diffusion XL
                    with specialized controlnets to ensure seamless integration between the original and 
                    newly generated areas.
                """)
            
        # Footer section
        with gr.Group(elem_classes="footer"):
            gr.HTML("""
                <p>Powered by <a href="https://huggingface.co/docs/diffusers/index" target="_blank">Diffusers</a> 
                and <a href="https://gradio.app/" target="_blank">Gradio</a> | 
                Using <a href="https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning" target="_blank">RealVisXL_V5.0_Lightning</a> Model</p>
            """)

    def use_output_as_input(output_image):
        """Sets the generated output as the new input image."""
        return gr.update(value=output_image)

    use_as_input_button.click(
        fn=use_output_as_input,
        inputs=[result],
        outputs=[input_image]
    )
    
    target_ratio.change(
        fn=preload_presets,
        inputs=[target_ratio, width_slider, height_slider],
        outputs=[width_slider, height_slider, settings_panel],
        queue=False
    )

    width_slider.change(
        fn=select_the_right_preset,
        inputs=[width_slider, height_slider],
        outputs=[target_ratio],
        queue=False
    )

    height_slider.change(
        fn=select_the_right_preset,
        inputs=[width_slider, height_slider],
        outputs=[target_ratio],
        queue=False
    )

    resize_option.change(
        fn=toggle_custom_resize_slider,
        inputs=[resize_option],
        outputs=[custom_resize_percentage],
        queue=False
    )
    
    run_button.click(  # Clear the result
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(  # Generate the new image
        fn=infer,
        inputs=[input_image, width_slider, height_slider, overlap_percentage, num_inference_steps,
                resize_option, custom_resize_percentage, prompt_input, alignment_dropdown,
                overlap_left, overlap_right, overlap_top, overlap_bottom],
        outputs=result,
    ).then(  # Update the history gallery
        fn=lambda x, history: update_history(x, history),
        inputs=[result, history_gallery],
        outputs=history_gallery,
    ).then(  # Show the "Use as Input Image" button
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=use_as_input_button,
    )

    prompt_input.submit(  # Clear the result
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(  # Generate the new image
        fn=infer,
        inputs=[input_image, width_slider, height_slider, overlap_percentage, num_inference_steps,
                resize_option, custom_resize_percentage, prompt_input, alignment_dropdown,
                overlap_left, overlap_right, overlap_top, overlap_bottom],
        outputs=result,
    ).then(  # Update the history gallery
        fn=lambda x, history: update_history(x, history),
        inputs=[result, history_gallery],
        outputs=history_gallery,
    ).then(  # Show the "Use as Input Image" button
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=use_as_input_button,
    )

    preview_button.click(
        fn=preview_image_and_mask,
        inputs=[input_image, width_slider, height_slider, overlap_percentage, resize_option, custom_resize_percentage, alignment_dropdown,
                overlap_left, overlap_right, overlap_top, overlap_bottom],
        outputs=preview_image,
        queue=False
    )

demo.queue(max_size=12).launch(server_name="127.0.0.1", server_port=7860, show_error=True)
