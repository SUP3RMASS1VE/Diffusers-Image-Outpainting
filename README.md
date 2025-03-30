# Diffusers Image Outpainting

![Screenshot 2025-03-30 201950](https://github.com/user-attachments/assets/9133933a-8398-4cb0-8fc3-44e481dbbae1)

A powerful web application that allows you to expand the boundaries of your images with AI-generated content, creating seamless extensions that match the original image style and context.

## Features

- **Intuitive Interface**: Easy-to-use controls for expanding your images in any direction
- **Flexible Canvas Sizing**: Preset aspect ratios (9:16, 16:9, 1:1) or custom dimensions
- **Smart Alignment**: Position your original image wherever you want on the new canvas
- **Customizable Generation**: Add prompts to guide the AI in creating contextually relevant expansions
- **Real-time Preview**: See exactly where new content will be generated before processing
- **History Gallery**: Keep track of your previous generations for comparison
- **Advanced Controls**: Fine-tune mask overlap, resize settings, and generation steps

## How It Works

This application uses the Stable Diffusion XL model with specialized ControlNet models to analyze your image and generate contextually appropriate outpainted regions. The process:

1. Upload an image you want to expand
2. Set your desired canvas size and image alignment
3. Add an optional text prompt to guide the generation
4. Preview how the image will be masked and positioned
5. Generate the outpainted result
6. Use the result as a new input to continue expanding if desired

## Technical Details

- Built with Diffusers library and Gradio
- Utilizes RealVisXL_V5.0_Lightning model for high-quality generation
- Implements a custom ControlNet Union approach for better boundary coherence
- Optimized for GPU acceleration with PyTorch

## Installation

```bash
# Clone this repository
git clone https://github.com/username/diffusers-image-outpaint.git
cd diffusers-image-outpaint

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Run the application
python app.py
```

The application will be available at http://localhost:7860 by default.

## Tips for Better Results

- Use images with clear boundaries and well-defined content
- Be specific in your prompts to guide the outpainting process
- Experiment with different mask overlap percentages for optimal blending
- Try different alignments to get the desired expansion direction
- For extending backgrounds, use simple descriptive prompts of the environment

## Examples

Check the `examples` directory for sample images and their outpainted results.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgements

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) for the core diffusion models
- [Gradio](https://gradio.app/) for the web interface framework
- [RealVisXL](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning) for the base diffusion model
