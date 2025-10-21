import os
import re
import uuid
import tempfile
import random
import csv
import datetime
import torch
import torchaudio
import soundfile as sf
import numpy as np
import gradio as gr
from cached_path import cached_path
from pathlib import Path

# -------------------------------
# F5-TTS imports
# -------------------------------
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT
from f5_tts.model.utils import seed_everything

# -------------------------------
# Model config
# -------------------------------
DEFAULT_MODEL_PATH = "hf://Pakorn2112/F5TTS-Hmong/model_159952.safetensors"
DEFAULT_REF_AUDIO_PATH = "src/f5_tts/infer/examples/thai_examples/ref_hmn.wav"
DEFAULT_REF_TEXT = "Kuv pom ib tug npauj npaim zoo nkauj heev li."
VOCAB_PATH = "./vocab/vocab.txt"
VOCAB_IPA_PATH = "./vocab/vocab_ipa.txt"

# -------------------------------
# Load model + vocoder
# -------------------------------
vocoder = load_vocoder()

def load_f5tts(ckpt_path, vocab_path=VOCAB_PATH, model_type="v1"):
    if model_type == "v1":
        cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2,
            text_dim=512, text_mask_padding=False,
            conv_layers=4, pe_attn_head=1,
        )
    else:
        cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2,
            text_dim=512, text_mask_padding=True,
            conv_layers=4, pe_attn_head=None,
        )
        vocab_path = VOCAB_IPA_PATH
    model = load_model(DiT, cfg, ckpt_path, vocab_file=vocab_path, use_ema=True)
    print(f"✅ Loaded F5-TTS model from {ckpt_path}")
    return model

try:
    f5tts_model = load_f5tts(str(cached_path(DEFAULT_MODEL_PATH)))
except Exception as e:
    print(f"❌ Error loading model: {e}")
    f5tts_model = None

# -------------------------------
# Text cleaning
# -------------------------------
def replace_numbers_with_hmong(text: str) -> str:
    number_map = {
        "0": "xoom", "1": "ib", "2": "ob", "3": "peb", "4": "plaub",
        "5": "tsib", "6": "rau", "7": "xya", "8": "yim", "9": "cuaj",
        "10": "kaum", "11": "kaum ib", "12": "kaum ob", "20": "nees nkaum",
        "30": "peb caug", "40": "plaub caug", "50": "tsib caug",
        "100": "ib puas", "1000": "ib txhiab"
    }
    def replace_match(match):
        number_str = match.group()
        return " ".join([number_map.get(c, c) for c in number_str])
    return re.sub(r"\d+(?:[.,]\d+)?", replace_match, text)

def process_hmong_repeat(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

# -------------------------------
# 🧾 Save user request
# -------------------------------
def log_user_request(ref_audio_path, ref_text, gen_text):
    """บันทึกข้อมูลผู้ใช้ที่ส่งข้อความเข้ามา"""
    log_file = "user_requests.csv"
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, ref_audio_path or "None", ref_text or "None", gen_text or "None"]

    header = ["timestamp", "ref_audio_path", "ref_text", "gen_text"]
    write_header = not os.path.exists(log_file)

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

# -------------------------------
# Core TTS
# -------------------------------
def tts_generate(gen_text, ref_audio=None, ref_text=None,
                 nfe_step=32, speed=0.9, cfg_strength=2.0,
                 cross_fade=0.15, remove_silence=True, seed=-1, use_ipa=False):

    if f5tts_model is None:
        return None, "❌ Model not loaded"

    if ref_audio is None or ref_audio == "":
        ref_audio_path = DEFAULT_REF_AUDIO_PATH
        ref_text = DEFAULT_REF_TEXT
    else:
        ref_audio_path = ref_audio

    # ✅ เก็บข้อมูลผู้ใช้
    log_user_request(ref_audio_path, ref_text, gen_text)

    if seed == -1:
        seed = random.randint(0, 999999)
    seed_everything(seed)

    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(ref_audio_path, ref_text or DEFAULT_REF_TEXT)
    gen_text_clean = process_hmong_repeat(replace_numbers_with_hmong(gen_text))

    try:
        final_wave, sr, _ = infer_process(
            ref_audio_proc, ref_text_proc, gen_text_clean,
            f5tts_model, vocoder,
            cross_fade_duration=cross_fade,
            nfe_step=nfe_step, speed=speed,
            cfg_strength=cfg_strength,
            set_max_chars=250, use_ipa=use_ipa,
        )
    except Exception as e:
        return None, f"❌ Inference failed: {e}"

    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, sr)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    output_name = f"tts_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(tempfile.gettempdir(), output_name)
    sf.write(output_path, final_wave, sr)
    return output_path, f"✅ Success (seed={seed})"

# -------------------------------
# Gradio UI
# -------------------------------
def ui_interface():
    with gr.Blocks(title="F5-TTS Hmong") as demo:
        gr.Markdown("""
        # 🗣️ F5-TTS-Hmong
        Text-to-Speech สำหรับภาษาม้ง (Hmong)  
        ใช้โมเดล **F5-TTS** (HPC-LocalVoice edition) ที่รองรับการอ้างอิงเสียงต้นแบบ (reference audio)
        """)

        with gr.Row():
            gen_text = gr.Textbox(label="📝 ข้อความที่ต้องการให้พูด (ภาษาม้ง)",
                                  placeholder="Nrog koj tham txog kev ncig teb chaws ...")
            ref_audio = gr.Audio(label="🔊 เสียงอ้างอิง (ถ้ามี)", type="filepath")
        ref_text = gr.Textbox(label="📖 ข้อความอ้างอิงของเสียง (เช่น คำพูดต้นแบบ)",
                              value=DEFAULT_REF_TEXT)

        with gr.Accordion("⚙️ ตัวเลือกเพิ่มเติม", open=False):
            nfe_step = gr.Slider(8, 64, 32, step=4, label="Diffusion Steps (nfe_step)")
            speed = gr.Slider(0.5, 1.5, 0.9, step=0.05, label="Speed")
            cfg_strength = gr.Slider(0.5, 3.0, 2.0, step=0.1, label="CFG Strength")
            cross_fade = gr.Slider(0.0, 0.3, 0.15, step=0.01, label="Cross-fade duration")
            remove_silence = gr.Checkbox(True, label="Remove silence at output")
            use_ipa = gr.Checkbox(False, label="Use IPA vocab")
            seed = gr.Number(-1, label="Random Seed (-1 = auto)")

        btn = gr.Button("🎧 สร้างเสียง (Generate)")
        audio_out = gr.Audio(label="🎵 ผลลัพธ์เสียง")
        status = gr.Markdown()

        # ตัวอย่างพร้อมเสียงอ้างอิง
        examples = [
            ["./src/f5_tts/infer/examples/thai_examples/ref_1.wav", "Peb yuav tsum siv dej kom txuag.", "Koj xav noj dab tsi?"],
            ["./src/f5_tts/infer/examples/thai_examples/ref_2.wav", "Kuv yuav nco ntsoov qhov chaw no.", "Koj nyob qhov twg?"],
            ["./src/f5_tts/infer/examples/thai_examples/ref_3.wav", "Kuv tsau lawm.", "Kuv noj mov tag lawm."],
            ["./src/f5_tts/infer/examples/thai_examples/update_80000_gen.wav", "Peb tsev neeg muaj 9 leeg, txhua tus sib hlub thiab sib pab.", "Kuv hu ua Paj Ntaub."],
        ]

        gr.Examples(
            examples=examples,
            inputs=[ref_audio, ref_text, gen_text],
            outputs=[audio_out, status],
            fn=tts_generate,
            cache_examples=False,
            label="🧩 ตัวอย่างพร้อมเสียงอ้างอิง"
        )

        btn.click(
            fn=tts_generate,
            inputs=[gen_text, ref_audio, ref_text, nfe_step, speed, cfg_strength,
                    cross_fade, remove_silence, seed, use_ipa],
            outputs=[audio_out, status]
        )

        gr.Markdown("💡 **Tips:** ทุกครั้งที่คุณสร้างเสียง ข้อความจะถูกเก็บไว้ในไฟล์ `user_requests.csv` เพื่อใช้วิเคราะห์และปรับปรุงระบบ")

    return demo

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    demo = ui_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860)


