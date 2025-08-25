from RealtimeSTT import AudioToTextRecorder
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='multiprocessing')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

max_new_tokens = 512
model_id = "meta-llama/Llama-3.2-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
)

def chat_completion_llama(prompt):
    # Apply chat template using official tokenizer support
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful voice AI assistant. Keep your answer short and don't use abbreviation and special tokens. Give your reply in hindi language only."},
            {"role": "user", "content": f"{prompt}"}
        ],
        return_tensors="pt",
        add_generation_prompt=True
    ).to(device)

    # Use TextStreamer for streaming output
    streamer = TextStreamer(tokenizer, skip_special_tokens=True)

    # Start generation with streaming
    _ = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id
    )

import threading
import asyncio
import queue
import time
import sounddevice as sd
import numpy as np
import torch
import warnings
import os
import sys
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from openai import OpenAI

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

client = OpenAI(
    base_url="https://api.sambanova.ai/v1",
    api_key="4a81daa9-5f3d-409b-9f30-ebedb379219a"
)

DEVICE = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
MODEL_1 = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(DEVICE)
MODEL_2 = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(DEVICE)

DESC_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_1.config.text_encoder._name_or_path)
DESCRIPTION = ("Rohit's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.")
DESC_INPUTS = DESC_TOKENIZER(DESCRIPTION, return_tensors="pt").to(DEVICE)
DESC_INPUT_IDS = DESC_INPUTS.input_ids
DESC_ATTN_MASK = DESC_INPUTS.attention_mask

SAMPLING_RATE = MODEL_1.config.sampling_rate
sd.default.latency = 'low'

MAX_CHUNK_CHARS = 10000

async def async_playback(queue1, queue2, done_event, latency_event):
    next_from_queue1 = True
    latency_measured = False

    while True:
        try:
            current_queue = queue1 if next_from_queue1 else queue2
            try:
                audio = await asyncio.wait_for(current_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                current_queue = queue2 if next_from_queue1 else queue1
                try:
                    audio = await asyncio.wait_for(current_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if done_event.is_set() and queue1.empty() and queue2.empty():
                        break
                    continue

            if latency_event and not latency_measured:
                latency_event.set()
                latency_measured = True

            sd.play(audio, samplerate=SAMPLING_RATE)
            sd.wait()

            next_from_queue1 = not (current_queue is queue1)

        except asyncio.CancelledError:
            break

    print("Playback finished.")

def tts_worker(text_queue, audio_queue, model, loop):
    async def put_audio(audio):
        await audio_queue.put(audio)

    while True:
        text = text_queue.get()
        if text is None:
            break

        prompt_inputs = TOKENIZER(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            speech_output = model.generate(
                input_ids=DESC_INPUT_IDS,
                attention_mask=DESC_ATTN_MASK,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask
            )
        audio = speech_output.cpu().numpy().squeeze()
        if audio.ndim > 1:
            audio = audio.flatten()
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        asyncio.run_coroutine_threadsafe(put_audio(audio), loop)

def stream_llm_to_chunks(prompt, q1, q2, small_chunk=20, large_chunk=60):
    buffer = ""
    toggle = 0
    max_queue_size=10
    chunk_sizes = [small_chunk, large_chunk]
    chunk_index = 0

    response = client.chat.completions.create(
        model="Meta-Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": "You are a Hindi-only assistant. Keep responses short."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    for chunk in response:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            buffer += delta.content
            print(delta.content, end="", flush=True)
            current_chunk_size = chunk_sizes[chunk_index % 2]

            while len(buffer) >= current_chunk_size:
                chunk_text = buffer[:current_chunk_size].strip()
                target_queue = q1 if toggle == 0 else q2

                while target_queue.qsize() >= max_queue_size:
                    time.sleep(0.01)

                target_queue.put(chunk_text)


                buffer = buffer[current_chunk_size:]
                toggle = 1 - toggle
                chunk_sizes[chunk_index % 2] = min(chunk_sizes[chunk_index % 2] * 2, MAX_CHUNK_CHARS)
                chunk_index += 1
    if buffer:
        (q1 if toggle == 0 else q2).put(buffer.strip())

    time.sleep(1.0)  # Prevent premature shutdown
    q1.put(None)
    q2.put(None)

def speak_from_prompt(prompt: str, small_chunk: int = 20, large_chunk: int = 40, measure_latency: bool = False):
    Q1, Q2 = queue.Queue(), queue.Queue()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    A1, A2 = asyncio.Queue(), asyncio.Queue()
    done_event = threading.Event()
    latency_event = threading.Event() if measure_latency else None

    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()

    tts_thread1 = threading.Thread(target=tts_worker, args=(Q1, A1, MODEL_1, loop))
    tts_thread2 = threading.Thread(target=tts_worker, args=(Q2, A2, MODEL_2, loop))
    tts_thread1.start()
    tts_thread2.start()

    start_time = time.time() if measure_latency else None
    playback_future = asyncio.run_coroutine_threadsafe(
        async_playback(A1, A2, done_event, latency_event), loop
    )

    stream_llm_to_chunks(prompt, Q1, Q2, small_chunk, large_chunk)
    tts_thread1.join()
    tts_thread2.join()

    done_event.set()
    playback_future.result()

    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join()

    if measure_latency and latency_event:
        latency_event.wait()
        latency = time.time() - start_time
        print(f"\nSpeech synthesis complete.\n⏱️ First audio latency: {latency:.2f} seconds")
    else:
        print("\nSpeech synthesis complete.")



if __name__ == '__main__':
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder(model="medium",device="cpu",print_transcription_time=True, allowed_latency_limit=50,language="hi",spinner=True)
    while True:
        # recorder.text(speak_from_prompt)
        recorder.text(chat_completion_llama)




