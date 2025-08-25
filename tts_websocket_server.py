from flask import Flask, request, Response, jsonify
import numpy as np
import soundfile as sf
import io
import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
from threading import Thread, Lock
import logging

app = Flask(__name__)

# Initialize model and tokenizer
model_name = "ai4bharat/indic-parler-tts"
torch_device = "cuda:0"  # Use "mps" for Mac or "cpu" if unavailable
torch_dtype = torch.bfloat16 if torch_device.startswith("cuda") else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name, attn_implementation="eager"
).to(torch_device, dtype=torch_dtype)

sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

# Thread lock for model inference
model_lock = Lock()

def generate_audio_stream(text, description, chunk_size_in_s=0.5, volume_boost=2.0):
    play_steps = int(frame_rate * chunk_size_in_s)
    streamer = ParlerTTSStreamer(model, device=torch_device, play_steps=play_steps)

    # Tokenization
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

    try:
        for new_audio in streamer:
            if new_audio.shape[0] == 0:
                break
            
            # Convert to numpy and apply volume boost
            audio_chunk = new_audio
            
            # Normalize and amplify audio
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 0:
                audio_chunk = (audio_chunk / max_val) * 0.99  # Normalize to [-0.99, 0.99]
                audio_chunk = audio_chunk * volume_boost  # Apply volume boost
            
            # Convert to WAV
            buffer = io.BytesIO()
            sf.write(buffer, audio_chunk, sampling_rate, format='WAV', subtype='PCM_16')
            wav_bytes = buffer.getvalue()
            
            yield wav_bytes

    finally:
        thread.join()
@app.route('/tts-stream', methods=['POST'])
def tts_stream():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request must be JSON"}), 400

    text = data.get('text')
    description = data.get('description')
    chunk_size_in_s = data.get('chunk_size_in_s', 0.5)

    if not text or not description:
        return jsonify({"error": "Missing 'text' or 'description' in request"}), 400

    try:
        with model_lock:
            return Response(
                generate_audio_stream(text, description, chunk_size_in_s),
                mimetype='audio/wav',
                headers={
                    'Content-Type': 'audio/wav',
                    'Cache-Control': 'no-cache',
                    'Transfer-Encoding': 'chunked'
                }
            )
    except Exception as e:
        logging.error(f"Error generating TTS: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)