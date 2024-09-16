import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from django.conf import settings


load_dotenv()

speech_config = speechsdk.SpeechConfig(subscription=os.getenv('SPEECH_KEY'),region=os.getenv('SPEECH_REGION'))
speech_config.speech_synthesis_voice_name = 'en-US-JaneNeural'

def text_to_speech(text):
    """
    Convert text to speech and return the URL of the audio file.
    
    :param text: The text to convert to speech.
    :return: The URL of the generated audio file.
    """
    # Path for saving audio file in the media folder
    audio_file_path = os.path.join(settings.MEDIA_ROOT, 'audio.wav')

    # Ensure the media folder exists
    if not os.path.exists(settings.MEDIA_ROOT):
        os.makedirs(settings.MEDIA_ROOT)

    # Set the output file to audio.wav in the media folder
    audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_file_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # Synthesize the speech
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized for text [{text}] and saved to {audio_file_path}.")
        # Construct and return the URL of the saved file
        audio_url = os.path.join(settings.MEDIA_URL, 'audio.wav')
        return audio_url
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print(f"Error details: {cancellation_details.error_details}")
                print("Did you set the speech resource key and region values?")
        return None