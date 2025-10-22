import ctypes
import numpy as np
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager

# --- 1. Load Library & Define Signatures ---

# Helper function to load and configure the C library
def load_library_and_signatures():
    try:
        lib_path = os.path.abspath('./nn_lib.so')
        lib = ctypes.CDLL(lib_path)
        print(f"Successfully loaded library from: {lib_path}")
    except OSError as e:
        print(f"FATAL ERROR: Could not load nn_lib.so.")
        print(f"Details: {e}")
        print("Make sure you compiled 'nn_lib.c' first:")
        print("  gcc -shared -o nn_lib.so -fPIC nn_lib.c -lm")
        exit()

    # C: void* create_network(int, int*, int, int, const char*, const char*)
    lib.create_network.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"), # int*
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p, # const char*
        ctypes.c_char_p  # const char*
    ]
    lib.create_network.restype = ctypes.c_void_p # void*

    # C: void train_step(void*, int, double*, double*, double)
    lib.train_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # double*
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # double*
        ctypes.c_double
    ]
    lib.train_step.restype = None # void

    # C: void predict(void*, int, double*, double*)
    lib.predict.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # double*
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")  # double*
    ]
    lib.predict.restype = None # void

    # C: void free_network(void*, int)
    lib.free_network.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.free_network.restype = None # void
    
    return lib

# Load the library when the script is imported
lib = load_library_and_signatures()


# --- 2. Training and Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # === ON STARTUP ===
    print("FastAPI startup: Training the model...")
    
    # Network parameters
    input_size = 1
    hidden_sizes_list = [4]
    hidden_sizes_arr = np.array(hidden_sizes_list, dtype=np.int32)
    num_hidden = len(hidden_sizes_list)
    output_size = 1
    total_layers = num_hidden + 1

    # Data
    train_input = np.array([
        [1.0], [2.0], [3.0], [4.0], [5.0], [-2.0]
    ], dtype=np.float64)
    train_target = np.array([
        [2.0], [4.0], [6.0], [8.0], [10.0], [-4.0]
    ], dtype=np.float64)
    num_samples = len(train_input)
    epochs = 2000
    learning_rate = 0.001

    # Create and train the model
    network_handle = lib.create_network(
        input_size, hidden_sizes_arr, num_hidden, output_size,
        b"leaky_relu", b"linear"
    )

    if not network_handle:
        print("FATAL ERROR: C function create_network failed to return a handle.")
        yield
        return # Exit the lifespan manager

    print(f"Training for {epochs} epochs...")
    for i in range(epochs):
        for j in range(num_samples):
            lib.train_step(
                network_handle, total_layers, 
                train_input[j], train_target[j], learning_rate
            )
        if i % 400 == 0:
             print(f"  ... training epoch {i}")
             
    print("Model training complete. API is ready.")

    # Store the trained model handle in the app's state
    app.state.network_handle = network_handle
    app.state.total_layers = total_layers
    app.state.output_size = output_size
    
    # Let the application run
    yield
    
    # === ON SHUTDOWN ===
    print("FastAPI shutdown: Freeing network memory...")
    lib.free_network(app.state.network_handle, app.state.total_layers)
    print("Memory freed. Goodbye.")

# --- 3. Create FastAPI App ---
app = FastAPI(lifespan=lifespan)


# --- 4. Define the API Endpoint ---

def _run_prediction(input_val: float) -> float:
    """Helper function to run a single prediction."""
    # Retrieve the trained model from the app state
    handle = app.state.network_handle
    total_layers = app.state.total_layers
    output_size = app.state.output_size

    # Prepare C-compatible arrays
    test_arr = np.array([input_val], dtype=np.float64)
    output_buffer = np.zeros(output_size, dtype=np.float64)
    
    # Call the C predict function
    lib.predict(handle, total_layers, test_arr, output_buffer)
    
    return output_buffer[0]


@app.get("/predict")
async def predict_unseen_data():
    """
    Runs the pre-trained C neural network on a fixed set of unseen data
    (2.5, 10.0, -3.0) and returns the predictions.
    """
    if not app.state.network_handle:
        return {"error": "Model is not initialized. Check server logs."}

    # Run predictions for the three unseen values
    pred_1 = _run_prediction(2.5)
    pred_2 = _run_prediction(10.0)
    pred_3 = _run_prediction(-3.0)

    # Format and return as JSON
    return {
        "predictions": [
            {"input": 2.5, "output": f"{pred_1:.4f}", "target": 5.0},
            {"input": 10.0, "output": f"{pred_2:.4f}", "target": 20.0},
            {"input": -3.0, "output": f"{pred_3:.4f}", "target": -6.0},
        ]
    }