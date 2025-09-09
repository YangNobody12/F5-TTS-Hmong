import random
import sys
import tempfile
import torchaudio
import soundfile as sf
from cached_path import cached_path
import argparse
import os
import uuid
import re
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import numpy as np
from collections import OrderedDict
from pathlib import Path

# Your existing imports
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from f5_tts.model import DiT
from f5_tts.model.utils import seed_everything
import torch
from f5_tts.utils.whisper_api import translate_inference, transribe_inference

# Model configuration
default_model_base = "D:\\tts_model\\all_model\\best_model\\model_159952.safetensors"
v2_model_base = "hf://Pakorn2112/F5TTS-Hmong/model_159952.safetensors"
vocab_base = "./vocab/vocab.txt"
vocab_ipa_base = "./vocab/vocab_ipa.txt"

# Global model instance
f5tts_model = None
vocoder = load_vocoder()


# Text cleaning functions (replacement for missing modules)
def replace_numbers_with_thai(text: str) -> str:
    """
    Replace numbers with Thai words (simplified version)
    """
    number_map = {
        "0": "xoom",
        "1": "ib",
        "2": "ob",
        "3": "peb",
        "4": "plaub",
        "5": "tsib",
        "6": "rau",
        "7": "xya",
        "8": "yim",
        "9": "cuaj",
        "10": "kaum",
        "11": "kaum ib",
        "12": "kaum ob",
        "20": "nees nkaum",
        "30": "peb caug",
        "40": "plaub caug",
        "50": "tsib caug",
        "100": "ib puas",
        "1000": "ib txhiab",
        "10000": "ib vam",
        "100000": "ib puas txhiab",
        "1000000": "ib lab",
    }

    def replace_match(match):
        number_str = match.group()
        result = []
        for char in number_str:
            if char in number_map:
                result.append(number_map[char])
            else:
                result.append(char)
        return " ".join(result)

    # Match numbers (including decimals)
    text = re.sub(r"\d+(?:[.,]\d+)?", replace_match, text)
    return text


def process_thai_repeat(text: str) -> str:
    """
    Process Thai repeated characters (simplified version)
    """
    # Basic repetition handling - remove excessive repetitions
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # Reduce 3+ repetitions to 2
    return text


def load_f5tts(ckpt_path, vocab_path=vocab_base, model_type="v1"):
    if model_type == "v1":
        F5TTS_model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            pe_attn_head=1,
        )
    elif model_type == "v2":
        F5TTS_model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=True,
            conv_layers=4,
            pe_attn_head=None,
        )
        vocab_path = "./vocab/vocab_ipa.txt"
    model = load_model(
        DiT, F5TTS_model_cfg, ckpt_path, vocab_file=vocab_path, use_ema=True
    )
    print(f"Loaded model from {ckpt_path}")
    return model


# Initialize model on startup
try:
    f5tts_model = load_f5tts(str(cached_path(default_model_base)))
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    f5tts_model = None

# FastAPI app
app = FastAPI(title="F5-TTS Hmong API", description="TTS API for Hmong language")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class TTSRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    ref_text: str
    gen_text: str
    remove_silence: bool = True
    cross_fade_duration: float = 0.15
    nfe_step: int = 32
    speed: float = 1.0
    cfg_strength: float = 2.0
    max_chars: int = 250
    seed: int = -1
    lang_process: str = "Default"


class MultiStyleSegment(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    style: str
    text: str


class MultiStyleTTSRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    segments: List[MultiStyleSegment]
    speech_types: Dict[str, Dict[str, str]]  # style -> {audio_path, ref_text}
    remove_silence: bool = True
    cross_fade_duration: float = 0.15
    nfe_step: int = 32
    lang_process: bool = False


class STTRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    translate: bool = False
    model: str = "large-v2"
    compute_type: str = "float16"
    target_language: str = "th"
    source_language: str = "Auto"


class ModelLoadRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_type: str = "Default"
    custom_model_path: Optional[str] = None


# Helper functions
def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location"""
    file_ext = os.path.splitext(upload_file.filename)[1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)

    with open(temp_file.name, "wb") as buffer:
        content = upload_file.file.read()
        buffer.write(content)

    return temp_file.name


def save_upload_file_path(path: str) -> str:
    """Save a file from given path to a temporary location"""
    path_obj = Path(path)
    file_ext = path_obj.suffix 

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)

    # อ่านไฟล์ต้นทางเป็นไบต์และเขียนลงไฟล์ชั่วคราว
    with open(path, "rb") as f:
        content = f.read()
    with open(temp_file.name, "wb") as buffer:
        buffer.write(content)

    return temp_file.name


def parse_speechtypes_text(text: str):
    """Parse multi-style text into segments"""
    segments = []
    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("{") and "}" in line:
            brace_end = line.find("}")
            style = line[1:brace_end].strip()
            text_content = line[brace_end + 1 :].strip()
            segments.append({"style": style, "text": text_content})
        else:
            # Default style if no style specified
            segments.append({"style": "ปกติ", "text": line})

    return segments


DEFAULT_REF_AUDIO_PATH = "src/f5_tts/infer/examples/thai_examples/ref_hmn.wav"
DEFAULT_REF_TEXT = "Kuv pom ib tug npauj npaim zoo nkauj heev li."


# # API endpoints
@app.post("/api/tts/generate")
async def generate_tts(
    ref_audio: Optional[UploadFile] = File(None),
    ref_text: str = Form(None),
    gen_text: str = Form(...),
    remove_silence: bool = Form(True),
    cross_fade_duration: float = Form(0.15),
    nfe_step: int = Form(32),
    speed: float = Form(0.6),
    cfg_strength: float = Form(2.0),
    max_chars: int = Form(250),
    seed: int = Form(-1),
    lang_process: str = Form("Default"),
):
    """Generate TTS from reference audio and text"""
    try:
        if f5tts_model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Save uploaded audio
        if ref_audio is None:
            print("Using default reference audio")
            ref_audio_path = save_upload_file_path(DEFAULT_REF_AUDIO_PATH)
            ref_text = DEFAULT_REF_TEXT
        else:
            ref_audio_path = save_upload_file(ref_audio)

        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)

        # Preprocess reference audio and text
        ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
            ref_audio_path, ref_text
        )

        # Clean generated text using our replacement functions
        gen_text_cleaned = process_thai_repeat(replace_numbers_with_thai(gen_text))

        # Generate audio
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio_processed,
            ref_text_processed,
            gen_text_cleaned,
            f5tts_model,
            vocoder,
            cross_fade_duration=float(cross_fade_duration),
            nfe_step=nfe_step,
            speed=speed,
            cfg_strength=cfg_strength,
            set_max_chars=max_chars,
            use_ipa=True if lang_process == "IPA" else False,
        )

        if remove_silence:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, final_wave, final_sample_rate)
                remove_silence_for_generated_wav(f.name)
                final_wave, _ = torchaudio.load(f.name)
            final_wave = final_wave.squeeze().cpu().numpy()

        # Save output audio
        output_filename = f"tts_output_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        sf.write(output_path, final_wave, final_sample_rate)

        # Cleanup
        os.unlink(ref_audio_path)

        return {
            "success": True,
            "audio_url": f"/download/{output_filename}",
            "seed": seed,
            "ref_text": ref_text_processed,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated audio files"""
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav", filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": f5tts_model is not None}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "F5-TTS Hmong API",
        "endpoints": {
            "/api/tts/generate": "Generate TTS from reference audio",
            "/api/tts/multistyle": "Generate multi-style TTS",
            "/api/stt/transcribe": "Convert speech to text",
            "/api/model/load": "Load different model version",
            "/api/health": "Health check",
        },
    }


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="F5-TTS API Server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run server on"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
