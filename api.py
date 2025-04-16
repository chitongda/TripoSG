import argparse
import os
import sys
import tempfile
import torch
import numpy as np
import trimesh
from PIL import Image
from typing import Any, Union
import io  # Added for BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask # Correct import for BackgroundTask
import uvicorn

# Add the project directory to the Python path to find modules
project_root = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(project_root) # This might not be needed if running with uvicorn from root
sys.path.append(os.path.join(project_root, "scripts")) # Ensure scripts dir is findable
# Ensure triposg module is findable
sys.path.append(project_root)


from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from image_process import prepare_image # Assuming image_process.py is in scripts/
from briarmbg import BriaRMBG

def run_inference(
    image_input: Image.Image,
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
) -> trimesh.Trimesh: # Changed return type hint for clarity

    if pipe is None or rmbg_net is None:
        raise RuntimeError("Models are not loaded.")

    # Ensure image is RGB
    if image_input.mode != 'RGB':
        print(f"Converting image from {image_input.mode} to RGB")
        image_input = image_input.convert('RGB')

    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
    print("Image prepared.")

    generator = torch.Generator(device=pipe.device).manual_seed(seed) # Use pipe.device

    outputs = pipe(
        image=img_pil,
    )

    # Ensure correct types for trimesh
    vertices = outputs[0].astype(np.float64) # Use float64 for trimesh compatibility
    faces = np.ascontiguousarray(outputs[1])

    # Check for degenerate faces (optional but good practice)
    if faces.shape[0] == 0:
        print("Warning: Generated mesh has no faces.")
        # Handle appropriately, maybe return an empty mesh or raise an error
        # For now, create an empty Trimesh
        return trimesh.Trimesh()


    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    print("Mesh created.")
    return mesh

async def generate_model(
    image: UploadFile = File(..., description="Input image file"),
    seed: int = Form(42, description="Random seed for generation"),
    num_inference_steps: int = Form(50, description="Number of inference steps", ge=1),
    guidance_scale: float = Form(7.0, description="Guidance scale", ge=0.0)
):
    # Define cleanup task function
    def cleanup(temp_file_path):
        try:
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")
        except OSError as e:
            print(f"Error cleaning up temporary file {temp_file_path}: {e}")

    output_path = None # Initialize output_path
    try:
        print(f"Received request for image: {image.filename}")
        # Read image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))

        # Run inference
        mesh = run_inference(
            image_input=pil_image,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        # Save mesh to a temporary file
        # Using 'with' ensures the file handle is closed, but delete=False keeps the file
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp_file:
            output_path = tmp_file.name # Store the path for cleanup
            print(f"Saving mesh to temporary file: {output_path}")
            mesh.export(output_path) # Export requires the file path


        # Return the file and schedule it for deletion after sending
        return FileResponse(
             output_path,
             media_type='model/gltf-binary',
             filename='output.glb',
             background=BackgroundTask(cleanup, output_path) # Pass path to cleanup task
        )

    except FileNotFoundError as e:
         print(f"Error: {e}")
         if output_path: # Attempt cleanup even if error occurred after file creation
             cleanup(output_path)
         raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
         print(f"Runtime Error: {e}")
         if output_path:
             cleanup(output_path)
         raise HTTPException(status_code=500, detail=f"Model inference error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        if output_path:
             cleanup(output_path)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- Main Execution ---
# For running directly with `python api.py` (optional)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    # Determine device based on availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Starting API...")
    # Models are loaded via the @app.on_event("startup") decorator when run with uvicorn

    uvicorn.run("api:app", host=args.host, port=args.port, reload=False) # Use string format for app 