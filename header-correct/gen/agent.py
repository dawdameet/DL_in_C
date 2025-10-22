import ctypes
import numpy as np
import os

# --- 1. Load the Shared Library ---
# Use os.path.abspath to get the full path to the .so file
lib_path = os.path.abspath('./nn_lib.so')
try:
    lib = ctypes.CDLL(lib_path)
except OSError as e:
    print(f"Error loading shared library: {e}")
    print("Make sure you compiled 'nn_lib.c' using the gcc command.")
    exit()

print(f"Successfully loaded library from: {lib_path}")


# --- 2. Define Function Signatures (Argtypes and Restype) ---
# This is the most important part. It tells ctypes how to
# translate Python types to C types.

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


# --- 3. Re-create the logic from your main() in Python ---

# Network parameters
input_size = 1
hidden_sizes_list = [4] # Your hidden layer sizes
hidden_sizes_arr = np.array(hidden_sizes_list, dtype=np.int32) # Must be numpy array
num_hidden = len(hidden_sizes_list)
output_size = 1
total_layers = num_hidden + 1

# Data (using numpy for easy C conversion)
train_input = np.array([
    [1.0], [2.0], [3.0], [4.0], [5.0], [-2.0]
], dtype=np.float64)
train_target = np.array([
    [2.0], [4.0], [6.0], [8.0], [10.0], [-4.0]
], dtype=np.float64)
num_samples = len(train_input)

epochs = 2000
learning_rate = 0.001

# --- 4. Call the C Functions ---

print("Creating network...")
# We use b"string" to create byte-strings, which C's char* expects
network_handle = lib.create_network(
    input_size,
    hidden_sizes_arr, 
    num_hidden,
    output_size,
    b"leaky_relu",
    b"linear"
)

if not network_handle:
    print("Failed to create network!")
    exit()

print(f"Training for {epochs} epochs...")
for i in range(epochs):
    epoch_error = 0.0
    for j in range(num_samples):
        # We pass the 1D slices (train_input[j]) to the C function
        lib.train_step(
            network_handle, 
            total_layers, 
            train_input[j], 
            train_target[j], 
            learning_rate
        )
    
    if i % 200 == 0:
        print(f"Epoch {i} complete")
        # Note: To get the error, you would need to write another
        # C API function to run forward() and return the error.
        # For simplicity, we just train.

print("\n--- Testing on Seen Data ---")
# Allocate a numpy array for the C function to write its output into
output_buffer = np.zeros(output_size, dtype=np.float64) 

for i in range(num_samples):
    lib.predict(network_handle, total_layers, train_input[i], output_buffer)
    print(f"Input: {train_input[i][0]:.1f} -> Output: {output_buffer[0]:.4f} (Target: {train_target[i][0]:.1f})")


print("\n--- Testing on Unseen Data ---")

# Test 1
test_val_1 = np.array([2.5], dtype=np.float64)
lib.predict(network_handle, total_layers, test_val_1, output_buffer)
print(f"Input: 2.5 -> Output: {output_buffer[0]:.4f} (Target: 5.0)")

# Test 2
test_val_2 = np.array([10.0], dtype=np.float64)
lib.predict(network_handle, total_layers, test_val_2, output_buffer)
print(f"Input: 10.0 -> Output: {output_buffer[0]:.4f} (Target: 20.0)")

# Test 3
test_val_3 = np.array([-3.0], dtype=np.float64)
lib.predict(network_handle, total_layers, test_val_3, output_buffer)
print(f"Input: -3.0 -> Output: {output_buffer[0]:.4f} (Target: -6.0)")

# --- 5. Clean Up C Memory ---
print("\nFreeing network memory...")
lib.free_network(network_handle, total_layers)
print("Done.")