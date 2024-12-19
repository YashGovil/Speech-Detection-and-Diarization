import os
import logging
from pydub import AudioSegment
from pyannote.audio import Pipeline
import cloudinary
import cloudinary.uploader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Cloudinary configuration
cloudinary.config(
    cloud_name='dszl5v7kg',
    api_key='383672918773879',
    api_secret='M9Y5zhdT_4Z8Ra75Wz2sfoY22T8',
    secure=True
)

# Initialize Hugging Face pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=os.getenv("HUGGINGFACE_AUTH_TOKEN")
)

def extract_and_upload_speaker_segments(input_file, diarization_result):
    audio = AudioSegment.from_wav(input_file)
    speaker_segments = {}

    for turn in diarization_result.itertracks(yield_label=True):
        start_ms = int(turn[0].start * 1000)
        end_ms = int(turn[0].end * 1000)
        speaker = turn[1]
        segment = audio[start_ms:end_ms]

        if speaker not in speaker_segments:
            speaker_segments[speaker] = segment
        else:
            speaker_segments[speaker] += segment

    uploaded_urls = []
    for speaker, merged_audio in speaker_segments.items():
        temp_file = f"{speaker}_merged.wav"
        merged_audio.export(temp_file, format="wav")

        try:
            upload_result = cloudinary.uploader.upload(
                temp_file,
                resource_type="video",
                folder="merged_speakers_output",
                public_id=speaker
            )
            uploaded_urls.append(upload_result['secure_url'])
        finally:
            os.remove(temp_file)

    return uploaded_urls

def process_audio_file(audio_file):
    diarization_result = pipeline(audio_file)
    return extract_and_upload_speaker_segments(audio_file, diarization_result)
