# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching. Support For Hmong language.

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/SWivid/F5-TTS)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06885)
[![lab](https://img.shields.io/badge/X--LANCE-Lab-grey?labelColor=lightgrey)](https://x-lance.sjtu.edu.cn/)
[![lab](https://img.shields.io/badge/Peng%20Cheng-Lab-grey?labelColor=lightgrey)](https://www.pcl.ac.cn)
<!-- <img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto"> -->

Text-to-Speech (TTS) ภาษาม้ง — เครื่องมือสร้างเสียงพูดจากข้อความด้วยเทคนิค Flow Matching ด้วยโมเดล F5-TTS

โมเดล Finetune : [Pakorn2112/F5TTS-Hmong](https://huggingface.co/Pakorn2112/F5TTS-Hmong) 

 - การอ่านข้อความยาวๆ หรือบางคำ ยังไม่ถูกต้อง

# การติดตั้ง
ก่อนเริ่มใช้งาน ต้องติดตั้ง:
 - Python (แนะนำเวอร์ชัน 3.10 ขึ้นไป)
 - [CUDA](https://developer.nvidia.com/cuda-downloads) แนะนำ CUDA version 11.8
 - [eSpeak NG](https://github.com/espeak-ng/espeak-ng)
```sh
git clone https://github.com/YangNobody12/F5-TTS-Hmong.git
cd F5-TTS-Hmong
python -m venv venv
call venv/scripts/activate
pip install git+https://github.com/VYNCX/F5-TTS-THAI.git

#จำเป็นต้องติดตั้งเพื่อใช้งานได้มีประสิทธิภาพกับ GPU
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
หรือ รันไฟล์ `install.bat` เพื่อติดตั้ง

# การใช้งาน
สามารถรันไฟล์ `app-webui.bat` เพื่อใช้งานได้ 
```sh
  python src/f5_tts/f5_tts_webui.py
```
หรือ 

```sh
  f5-tts_webui
```

คำแนะนำ :
- สามารถตั้งค่า "ตัวอักษรสูงสุดต่อส่วน" หรือ max_chars เพื่อลดความผิดพลาดการอ่าน แต่ความเร็วในการสร้างจะช้าลง สามารถปรับลด NFE Step เพื่อเพิ่มความเร็วได้.
- อย่าลืมเว้นวรรคประโยคเพื่อให้สามารถแบ่งส่วนในการสร้างได้.
- สำหรับ ref_text หรือ ข้อความตันฉบับ แนะนำให้ใช้เป็นภาษาม้งหรือคำอ่านภาษาม้งสำหรับเสียงภาษาอื่น เพื่อให้การอ่านภาษาม้งดีขึ้น เช่น Good Morning > กู้ดมอร์นิ่ง.
- สำหรับเสียงต้นแบบ ควรใช้ความยาวไม่เกิน 8 วินาที ถ้าเป็นไปได้ห้ามมีเสียงรบกวน.
- สามารถปรับลดความเร็ว เพื่อให้การอ่านคำดีขึ้นได้ เช่น ความเร็ว 0.8-0.9 เพื่อลดการอ่านผิดหรือคำขาดหาย แต่ลดมากไปอาจมีเสียงต้นฉบับแทรกเข้ามา.
  
  <details><summary>ตัวอย่าง WebUI</summary>
  
   - Text To Speech
   ![Example_Gradio#3](https://github.com/user-attachments/assets/9fd6bf42-3c34-41aa-8f88-3f7ea191e4f0)
  
   - Multi Speech
   ![Example_Gradio#4](https://github.com/user-attachments/assets/fc57b2d0-bef9-4454-94c3-b72ca2551265)
 
  

# ตัวอย่างเสียง
[demo ](https://demot2shmong.yangnobody.com/) 


# อ้างอิง

- [F5-TTS](https://github.com/SWivid/F5-TTS)
- [F5-TTS-THAI](https://github.com/VYNCX/F5-TTS-THAI)










