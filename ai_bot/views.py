from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import io
from pydub import AudioSegment
from .response_handler import chatbot_response
from .tts_model import text_to_speech
from django.conf import settings
from django.http import JsonResponse
import logging
import os

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and processor, move the model to the GPU
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device)
model.config.forced_decoder_ids = None

class TranscribeAudio(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        # Get the uploaded audio file from the request
        audio_file = request.FILES['file']

        # Read the webm audio file
        audio_bytes = audio_file.read()

        # Convert webm to wav using pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        audio = audio.set_frame_rate(16000)

        # Export the audio to a bytes buffer in wav format
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)  # Reset buffer position

        # Load the wav audio with librosa
        audio_data, sampling_rate = librosa.load(wav_io, sr=16000)

        # Convert the audio data to tensor
        audio_tensor = torch.tensor(audio_data).unsqueeze(0)  # Add batch dimension

        # Preprocess the audio to extract features, set language to English ('en')
        inputs = processor(audio_tensor.squeeze(), return_tensors="pt", sampling_rate=sampling_rate, language="en")
        inputs = {key: inputs[key].to(device) for key in inputs}

        # Use the model's generate method to produce predictions
        with torch.no_grad():
            predicted_ids = model.generate(inputs["input_features"])

        # Decode predicted token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcribed_text = ' '.join(transcription)
        print(transcribed_text)

        # Get chatbot response based on the transcription
        response = chatbot_response(transcribed_text)
        print(response)

        # Generate the audio file from the chatbot response using TTS
        audio_file_path = text_to_speech(response)

        if audio_file_path:
            logger.info(f"Generated audio file at: {audio_file_path}")

            # Construct the audio URL using MEDIA_URL
            audio_url = os.path.join(settings.MEDIA_URL, 'audio.wav')

            # Return the audio URL and transcription in a JSON response
            return JsonResponse({
                "transcription": transcribed_text,
                "audio_url": audio_url
            })

        # If audio generation fails, return an error response
        logger.error("Failed to generate audio file.")
        return Response({"transcription": transcribed_text, "error": "Failed to generate audio."}, status=500)
