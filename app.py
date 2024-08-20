from flask import Flask, request
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import torchaudio

app = Flask(__name__)

# Load the pre-trained Whisper large-v3 model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Get the audio file from the request
    audio_file = request.files['audio']

    # Load and preprocess the audio
    input_audio, sample_rate = torchaudio.load(audio_file)
    input_values = processor(audio=input_audio, sampling_rate=sample_rate, return_tensors="pt").input_values

    # Generate the text transcript
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_values, max_length=500, num_beams=1, early_stopping=True, return_dict_in_generate=True, output_scores=False, output_hidden_states=False)[0]
    text = processor.decode(output_ids[0], skip_special_tokens=True)

    return {'text': text}

if __name__ == '__main__':
    app.run()
