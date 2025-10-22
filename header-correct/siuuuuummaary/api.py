import ctypes
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. Load C Library & Define Signatures ---
try:
    lib_path = os.path.abspath('./summarize_lib.so')
    lib = ctypes.CDLL(lib_path)
    print(f"Successfully loaded library from: {lib_path}")
except OSError as e:
    print(f"FATAL ERROR: Could not load summarize_lib.so.")
    print(f"Details: {e}")
    print("Make sure you compiled 'summarize_lib.c' first:")
    print("  gcc -shared -o summarize_lib.so -fPIC summarize_lib.c -lm -fopenmp")
    exit()

# C: int summarize_text(const char*, const char*, char*, int, char*, int)
lib.summarize_text.argtypes = [
    ctypes.c_char_p,        # const char *text
    ctypes.c_char_p,        # const char *title
    ctypes.c_char_p,        # char *stop_content (strtok modifies it)
    ctypes.c_int,           # int n_summary
    ctypes.c_char_p,        # char *output_buffer
    ctypes.c_int            # int buffer_size
]
lib.summarize_text.restype = ctypes.c_int # Returns 0 or -1

# --- 2. FastAPI App & Pydantic Model ---

app = FastAPI()

class SummarizeRequest(BaseModel):
    text: str
    title: str = ""
    stopwords: str = "" # Send stopwords as a single string, e.g. "a\nan\nthe"
    n_summary: int = 5

# --- 3. API Endpoint ---
@app.post("/summarize")
async def run_summarization(request: SummarizeRequest):
    """
    Summarizes the input text using the C library.
    """
    # 1. Prepare C inputs
    input_text_c = request.text.encode('utf-8')
    title_c = request.title.encode('utf-8')
    
    # We must pass a *mutable* string for stopwords, as strtok modifies it
    stopwords_c = ctypes.create_string_buffer(request.stopwords.encode('utf-8'))
    
    n_summary_c = ctypes.c_int(request.n_summary)

    # 2. Prepare C output buffer
    # Let's make the buffer the same size as the input text + 1 for null terminator.
    # The summary can never be larger than the original text.
    buffer_size = len(input_text_c) + 1
    output_buffer_c = ctypes.create_string_buffer(buffer_size)

    # 3. Call C function
    result_code = lib.summarize_text(
        input_text_c,
        title_c,
        stopwords_c,
        n_summary_c,
        output_buffer_c,
        buffer_size
    )

    # 4. Check results
    if result_code == -1:
        raise HTTPException(
            status_code=500, 
            detail="Summarization failed: Output buffer overflow."
        )

    # 5. Decode and format the output
    summary_string = output_buffer_c.value.decode('utf-8')
    
    # Split the newline-separated string into a clean JSON list
    summary_list = [
        s.strip() for s in summary_string.strip().split('\n') 
        if s.strip() # Remove any empty lines
    ]

    return {
        "summary": summary_list,
        "sentences_in_summary": len(summary_list),
        "sentences_requested": request.n_summary,
    }

@app.get("/")
async def root():
    return {"message": "Extractive Summarizer API is running. POST to /summarize"}