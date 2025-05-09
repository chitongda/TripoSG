import argparse
import os
import sys
import tempfile
import torch
import numpy as np
import trimesh
from PIL import Image
import typing as T # Use typing alias
import io
import uuid # Added for task IDs
import asyncio # Added for background tasks
# import onnxruntime # Removed ONNX runtime import
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask # Correct import for BackgroundTask
import uvicorn

# Add the project directory to the Python path to find modules
project_root = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(project_root) # This might not be needed if running with uvicorn from root
sys.path.append(os.path.join(project_root, "scripts")) # Ensure scripts dir is findable
# Ensure triposg module is findable (assuming it's installable or in root)
# sys.path.append(project_root) # Maybe not needed if installed
# Ensure briarmbg is findable (assuming it's installable or in root/scripts)
# sys.path.append(os.path.join(project_root, "briarmbg")) # Example if needed


from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from image_process import prepare_image # Assuming image_process.py is in scripts/
from briarmbg import BriaRMBG # Import PyTorch RMBG model

# --- ONNX Wrapper REMOVED ---
# class ONNXCallableWrapper:
# ... (wrapper code removed) ...
# --- End ONNX Wrapper ---


# Global variables for models and task statuses
pipe: T.Optional[TripoSGPipeline] = None
rmbg_net: T.Optional[BriaRMBG] = None # Use BriaRMBG type hint
task_statuses: T.Dict[str, T.Dict[str, T.Any]] = {} # Stores task status and results

# --- Model Loading ---
def load_models():
    """Loads the TripoSR and RMBG models."""
    global pipe, rmbg_net
    print("Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 # Force float32 for stability
    print(f"Using device: {device}, dtype: {dtype}") # Log the chosen dtype

    try:
        # Determine the path to the downloaded weights
        # Assumes download_models.py was run from the project root
        model_load_path = os.path.join(project_root, "pretrained_weights", "TripoSG")
        if not os.path.isdir(model_load_path):
            raise FileNotFoundError(f"TripoSR model not found at {model_load_path}. Please run download_models.py first.")

        print(f"Loading TripoSR model from local path: {model_load_path}")
        # Load first, then move and cast dtype
        pipe = TripoSGPipeline.from_pretrained(
            model_load_path # Removed torch_dtype=dtype here
        )
        pipe.to(device, dtype=dtype) # Move and set dtype here
        print("TripoSR model loaded.")

        # Load RMBG model from its directory (PyTorch version)
        rmbg_model_dir = os.path.join(project_root, 'pretrained_weights', 'RMBG-1.4')
        if not os.path.isdir(rmbg_model_dir):
             raise FileNotFoundError(f"RMBG model directory not found at {rmbg_model_dir}. Please ensure it exists.")
        print(f"Loading RMBG PyTorch model from: {rmbg_model_dir}")

        # Load BriaRMBG model using its from_pretrained method
        rmbg_net = BriaRMBG.from_pretrained(rmbg_model_dir)
        rmbg_net.to(device, dtype=dtype) # Move and set dtype here
        rmbg_net.eval() # Set to evaluation mode
        print("RMBG model loaded.")

    except ImportError as e:
        print(f"Error loading models: {e}. Please ensure all dependencies are installed.")
        raise RuntimeError(f"Failed to import necessary libraries: {e}") from e
    except FileNotFoundError as e:
         print(f"Model file error: {e}")
         raise RuntimeError(f"Model file error: {e}") from e
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e

# --- FastAPI App ---
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Load models when the application starts."""
    load_models()

# --- Helper Functions ---

def cleanup_temp_file(temp_file_path: str):
    """Safely removes a temporary file."""
    try:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")
        else:
            print(f"Temporary file already removed: {temp_file_path}")
    except OSError as e:
        print(f"Error cleaning up temporary file {temp_file_path}: {e}")

async def _run_generation_task(
    task_id: str,
    image_contents: bytes,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
):
    """The actual generation process, run in the background."""
    output_path = None
    try:
        task_statuses[task_id] = {"status": "running"}
        print(f"Task {task_id}: Starting generation.")

        pil_image = Image.open(io.BytesIO(image_contents))

        # Run inference (ensure run_inference is defined or integrate logic here)
        mesh = run_inference( # Assuming run_inference accesses global pipe/rmbg_net
            image_input=pil_image,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        # Save mesh to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp_file:
            output_path = tmp_file.name
            print(f"Task {task_id}: Saving mesh to {output_path}")
            mesh.export(output_path)

        # Update status to completed
        task_statuses[task_id] = {"status": "completed", "result_path": output_path}
        print(f"Task {task_id}: Completed successfully.")

    except FileNotFoundError as e:
        print(f"Task {task_id}: Error - {e}")
        task_statuses[task_id] = {"status": "failed", "error": str(e)}
        if output_path: cleanup_temp_file(output_path)
    except RuntimeError as e:
        print(f"Task {task_id}: Runtime Error - {e}")
        task_statuses[task_id] = {"status": "failed", "error": f"Model inference error: {e}"}
        if output_path: cleanup_temp_file(output_path)
    except Exception as e:
        print(f"Task {task_id}: Unexpected Error - {e}")
        import traceback
        traceback.print_exc()
        task_statuses[task_id] = {"status": "failed", "error": f"An unexpected error occurred: {e}"}
        if output_path: cleanup_temp_file(output_path)


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

    # --- Modification Start: Save PIL image to temp file ---
    temp_image_path = None
    try:
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
            temp_image_path = tmp_img_file.name
            image_input.save(temp_image_path, format='PNG') # Save the PIL image
            print(f"Saved input image temporarily to {temp_image_path}")

        # Pass the path to prepare_image
        img_pil = prepare_image(temp_image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
        print("Image prepared from temporary file.")

    finally:
        # Clean up the temporary image file
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                print(f"Cleaned up temporary image file: {temp_image_path}")
            except OSError as e:
                print(f"Error cleaning up temporary image file {temp_image_path}: {e}")
    # --- Modification End ---

    generator = torch.Generator(device=pipe.device).manual_seed(seed) # Use pipe.device

    # Run the pipeline
    pipeline_output = pipe(
        image=img_pil,
        # Add generator and other relevant parameters if needed
        # generator=generator, # Example if seed control per-request is desired
        # num_inference_steps=num_inference_steps, # Example
        # guidance_scale=guidance_scale, # Example
    )

    # Extract mesh data from the output object
    # Assuming the structure is similar to app.py: output.samples[0]
    if not hasattr(pipeline_output, 'samples') or not pipeline_output.samples:
        raise RuntimeError("Pipeline output does not contain 'samples' or it is empty.")
    
    mesh_data = pipeline_output.samples[0]
    
    # Ensure correct types for trimesh
    # Access vertices and faces from the extracted mesh_data
    if not isinstance(mesh_data, (list, tuple)) or len(mesh_data) < 2:
        raise RuntimeError(f"Unexpected format for mesh data inside samples. Expected list/tuple of length 2, got: {type(mesh_data)}")
        
    vertices_data = mesh_data[0]
    faces_data = mesh_data[1]

    # Check types before calling methods
    if not hasattr(vertices_data, 'astype'):
         raise TypeError(f"Vertices data type ({type(vertices_data)}) does not have 'astype' method.")
    if not isinstance(faces_data, (np.ndarray, torch.Tensor)):
         raise TypeError(f"Faces data type ({type(faces_data)}) is not suitable for np.ascontiguousarray.")

    vertices = vertices_data.astype(np.float64) # Use float64 for trimesh compatibility
    faces = np.ascontiguousarray(faces_data)

    # Check for degenerate faces (optional but good practice)
    if faces.shape[0] == 0:
        print("Warning: Generated mesh has no faces.")
        # Handle appropriately, maybe return an empty mesh or raise an error
        # For now, create an empty Trimesh
        return trimesh.Trimesh()


    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    print("Mesh created.")
    return mesh

# --- API Endpoints ---

@app.post("/generate/", status_code=202) # Use 202 Accepted for async tasks
async def start_generation(
    background_tasks: BackgroundTasks, # Use BackgroundTasks for dependency injection
    image: UploadFile = File(..., description="Input image file"),
    seed: int = Form(42, description="Random seed for generation"),
    num_inference_steps: int = Form(50, description="Number of inference steps", ge=1),
    guidance_scale: float = Form(7.0, description="Guidance scale", ge=0.0)
):
    """
    Accepts an image and parameters, starts the 3D model generation task
    in the background, and returns a task ID.
    """
    if pipe is None or rmbg_net is None:
         # This check might be redundant if startup guarantees loading, but good practice
         raise HTTPException(status_code=503, detail="Models are not loaded or still loading.")

    task_id = str(uuid.uuid4())
    print(f"Received generation request, assigning Task ID: {task_id}")

    # Read image content immediately
    try:
        image_contents = await image.read()
        # Quick validation if it's an image (optional, PIL will raise later if not)
        _ = Image.open(io.BytesIO(image_contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    finally:
        await image.close() # Ensure file handle is closed

    # Store initial status
    task_statuses[task_id] = {"status": "pending"}

    # Add the generation task to run in the background
    background_tasks.add_task(
        _run_generation_task,
        task_id,
        image_contents,
        seed,
        num_inference_steps,
        guidance_scale,
    )

    return {"task_id": task_id, "status": "pending"}

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Retrieves the status of a generation task by its ID.
    Possible statuses: pending, running, completed, failed.
    If failed, an 'error' field will be present.
    """
    status_info = task_statuses.get(task_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Task not found")
    return status_info

@app.get("/result/{task_id}")
async def get_task_result(task_id: str):
    """
    Retrieves the generated 3D model (.glb file) for a completed task.
    Deletes the temporary file after sending the response.
    """
    status_info = task_statuses.get(task_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Task not found")

    if status_info["status"] == "pending" or status_info["status"] == "running":
        raise HTTPException(status_code=400, detail=f"Task is still {status_info['status']}")
    elif status_info["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Task failed: {status_info.get('error', 'Unknown error')}")
    elif status_info["status"] == "completed":
        result_path = status_info.get("result_path")
        if not result_path or not os.path.exists(result_path):
             # Handle case where path is missing or file got deleted somehow
             print(f"Error: Result file path missing or file not found for task {task_id}")
             task_statuses[task_id] = {"status": "failed", "error": "Result file not found."} # Update status
             raise HTTPException(status_code=500, detail="Result file not found.")

        # Return the file and schedule cleanup
        return FileResponse(
             result_path,
             media_type='model/gltf-binary',
             filename=f'{task_id}_output.glb', # Use task_id in filename
             background=BackgroundTask(cleanup_temp_file, result_path) # Use the helper
        )
    else:
        # Should not happen with defined statuses
        raise HTTPException(status_code=500, detail="Unknown task status")

# Remove the old generate_model function if it exists (it's replaced by start_generation)
# The original generate_model logic is now in _run_generation_task

# --- Main Execution ---
# For running directly with `python api.py` (optional)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # Determine device based on availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Starting API service...") # Changed message slightly
    # Models are loaded via the @app.on_event("startup") decorator when run with uvicorn

    uvicorn.run("api:app", host=args.host, port=args.port, reload=False) # Use string format for app 