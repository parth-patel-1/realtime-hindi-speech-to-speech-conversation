import threading
import queue
import time
import sounddevice as sd
import numpy as np
import os
import warnings
import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from openai import OpenAI
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
# === YOUR API CLIENT ===
client = OpenAI(
    base_url="https://api.sambanova.ai/v1",
    api_key="4a81daa9-5f3d-409b-9f30-ebedb379219a"
)

# === Device & Models ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TTS setup
TOKENIZER = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
MODEL_1   = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(DEVICE)
MODEL_2   = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(DEVICE)

# Voice conditioning
DESC_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_1.config.text_encoder._name_or_path)
DESCRIPTION    = "Rohit's voice is monotone yet slightly fast in delivery, with minimal background noise."
DESC_INPUTS    = DESC_TOKENIZER(DESCRIPTION, return_tensors="pt").to(DEVICE)
DESC_INPUT_IDS = DESC_INPUTS.input_ids
DESC_ATTN_MASK = DESC_INPUTS.attention_mask

SAMPLING_RATE = MODEL_1.config.sampling_rate
sd.default.latency = 'low'

# === TTS Worker ===
def tts_worker(text_queue, audio_dict, model, audio_events):
    while True:
        item = text_queue.get()
        if item is None:
            break
        seq_id, chunk, start_time = item

        inp = TOKENIZER(chunk, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            aud = model.generate(
                input_ids=DESC_INPUT_IDS,
                attention_mask=DESC_ATTN_MASK,
                prompt_input_ids=inp.input_ids,
                prompt_attention_mask=inp.attention_mask
            )
        audio = aud.cpu().numpy().squeeze()
        if audio.ndim>1: audio = audio.flatten()
        audio = audio.astype(np.float32)
        audio /= np.max(np.abs(audio)) or 1.0

        audio_dict[seq_id] = (audio, start_time)
        audio_events[seq_id].set()

# === Playback Thread ===
def playback_thread_fn(audio_dict, audio_events, chunk_ids, done_stream):
    next_seq = 0
    while True:
        while next_seq >= len(chunk_ids):
            if done_stream.is_set(): break
            time.sleep(0.01)
        if next_seq >= len(chunk_ids) and done_stream.is_set():
            break

        audio_events[next_seq].wait()
        audio, st = audio_dict[next_seq]
        if next_seq==0:
            latency = time.time() - st
            print(f"\n Latency: {latency:.2f}s")
        sd.play(audio, samplerate=SAMPLING_RATE)
        sd.wait()
        next_seq += 1

    # print("All done.")

# === Streaming + Dynamic Chunking + TTS ===
def stream_llm_tts(prompt, initial_size=30, max_size=1000):
    text_q1, text_q2 = queue.Queue(), queue.Queue()
    audio_dict, audio_events = {}, {}
    chunk_ids = []
    lock = threading.Lock()
    done_stream = threading.Event()

    # start TTS threads
    t1 = threading.Thread(target=tts_worker, args=(text_q1, audio_dict, MODEL_1, audio_events), daemon=True)
    t2 = threading.Thread(target=tts_worker, args=(text_q2, audio_dict, MODEL_2, audio_events), daemon=True)
    t1.start(); t2.start()
    # start playback
    pb = threading.Thread(target=playback_thread_fn, args=(audio_dict, audio_events, chunk_ids, done_stream), daemon=True)
    pb.start()

    buffer = ""
    toggle = 0
    sizes  = [initial_size, initial_size]

    response = client.chat.completions.create(
        model="Meta-Llama-3.2-3B-Instruct",
        messages=[
            {"role":"system", "content":"You are a Hindi-only assistant. Keep responses short."},
            {"role":"user",   "content":prompt}
        ],
        stream=True
    )

    # stream tokens
    for chunk in response:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            # print text live
            print(delta.content, end="", flush=True)

            buffer += delta.content
            # emit full chunks
            while len(buffer) >= sizes[toggle % 2]:
                txt = buffer[:sizes[toggle % 2]].strip()
                buffer = buffer[sizes[toggle % 2]:]

                with lock:
                    seq = len(chunk_ids)
                    chunk_ids.append(seq)
                    audio_events[seq] = threading.Event()

                # dispatch chunk
                (text_q1 if seq % 2 == 0 else text_q2).put((seq, txt, time.time()))

                # update sizes
                sizes[toggle % 2] = min(sizes[toggle % 2] * 2, max_size)
                toggle += 1

    # leftover
    if buffer.strip():
        print(buffer, end="", flush=True)
        with lock:
            seq = len(chunk_ids)
            chunk_ids.append(seq)
            audio_events[seq] = threading.Event()
        (text_q1 if seq % 2 == 0 else text_q2).put((seq, buffer.strip(), time.time()))

    # signal end
    text_q1.put(None); text_q2.put(None)
    done_stream.set()

    t1.join(); t2.join(); pb.join()


if __name__ == "__main__":
    from RealtimeSTT import AudioToTextRecorder
    print("Wait until it says 'speak now'")
    import time
    
    recorder = AudioToTextRecorder(model="medium",print_transcription_time=True,language="hi",spinner=False)
        
    while True:
        # recorder.text(speak_from_prompt)
        recorder.text(stream_llm_tts)
        #small, large
        