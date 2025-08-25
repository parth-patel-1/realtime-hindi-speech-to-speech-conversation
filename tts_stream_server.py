import torch
import io
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
from threading import Thread
from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse
import soundfile as sf

# Model setup
torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch_device == "cuda:0" else torch.float32
model_name = "ai4bharat/indic-parler-tts"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name, attn_implementation="eager"
).to(torch_device, dtype=torch_dtype)

sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

app = FastAPI()

def synthesize_stream(text, description, play_steps_in_s=0.5):
    play_steps = int(frame_rate * play_steps_in_s)
    streamer = ParlerTTSStreamer(model, device=torch_device, play_steps=play_steps)

    inputs = tokenizer(description, return_tensors="pt").to(torch_device)
    prompt = tokenizer(text, return_tensors="pt").to(torch_device)

    generation_kwargs = dict(
        input_ids=inputs.input_ids,
        prompt_input_ids=prompt.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_attention_mask=prompt.attention_mask,
        streamer=streamer,
        do_sample=True,
        temperature=1.0,
        min_new_tokens=10,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    audio_chunks = []
    for chunk in streamer:
        if chunk.shape[0] == 0:
            break
        audio_chunks.append(chunk)

    full_audio = torch.cat(audio_chunks).cpu().numpy()
    return full_audio

@app.post("/tts-stream")
async def tts_stream(text: str = Form(...), description: str = Form(...)):
    audio_np = synthesize_stream(text, description)

    # Save to in-memory buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, samplerate=sampling_rate, format='WAV')
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")
