import os
import warnings
import logging
from diffusers import logging as diffusers_logging

# Suppress all diffusers logging
diffusers_logging.set_verbosity_error()

# Set Python logging level to ERROR
logging.basicConfig(level=logging.ERROR)

# Set environment variable to disable safetensors warnings
os.environ["DIFFUSERS_NO_SAFETENSORS_WARNINGS"] = "1"

# Filter warnings about fp16 and non-fp16 filenames
warnings.filterwarnings("ignore", message=".*mixture of fp16 and non-fp16 filenames.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Monkey patch the safetensors warning method in diffusers
try:
    # This is a more aggressive approach - disable the method that issues the warning
    from diffusers.utils import safetensors_utils
    
    # Get the original method
    original_check_fp16_safetensors = None
    if hasattr(safetensors_utils, "check_fp16_safetensors"):
        original_check_fp16_safetensors = safetensors_utils.check_fp16_safetensors
        
        # Replace it with a version that doesn't warn
        def silent_check_fp16_safetensors(*args, **kwargs):
            if original_check_fp16_safetensors:
                # Call the original but suppress any warnings it might produce
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return original_check_fp16_safetensors(*args, **kwargs)
            return None
        
        # Apply the monkey patch
        safetensors_utils.check_fp16_safetensors = silent_check_fp16_safetensors
except Exception:
    # If anything goes wrong, just continue without the monkey patch
    pass 