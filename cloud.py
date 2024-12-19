import os
import logging
from pydub import AudioSegment
from pyannote.audio import Pipeline
from transformers import pipeline as transformers_pipeline
import cloudinary
import cloudinary.uploader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Cloudinary configuration using environment variables
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)

# Initialize Hugging Face pipelines
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=os.getenv("HUGGINGFACE_AUTH_TOKEN")
)
transcription_pipeline = transformers_pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Ensure output folders exist
MERGED_SPEAKERS_OUTPUT_FOLDER = "merged_speakers_output"
OUTPUT_TRANSCRIPTION_FOLDER = "output_transcription"
os.makedirs(MERGED_SPEAKERS_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_TRANSCRIPTION_FOLDER, exist_ok=True)

def extract_and_upload_speaker_segments(input_file, diarization_result):
    audio = AudioSegment.from_wav(input_file)
    speaker_segments = {}

    # Split and merge audio segments by speaker
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
    diarized_files = []

    for index, (speaker, merged_audio) in enumerate(speaker_segments.items()):
        temp_file = os.path.join(MERGED_SPEAKERS_OUTPUT_FOLDER, f"{index}_merged.wav")
        merged_audio.export(temp_file, format="wav")
        diarized_files.append(temp_file)

        try:
            upload_result = cloudinary.uploader.upload(
                temp_file,
                resource_type="video",
                folder="merged_speakers_output",
                public_id=f"speaker_{index}_merged"
            )
            uploaded_urls.append(upload_result['secure_url'])
            logging.info(f"Uploaded {temp_file} to Cloudinary.")
        except Exception as e:
            logging.error(f"Failed to upload {temp_file}: {e}")

    return uploaded_urls, diarized_files

def extract_transcription(audio_files, transcription_output_folder):
    transcriptions = {}
    for audio_file in audio_files:
        try:
            transcription_result = transcription_pipeline(audio_file)
            transcription_text = transcription_result['text']

            output_file = os.path.join(transcription_output_folder, f"{os.path.basename(audio_file)}.txt")
            with open(output_file, "w") as f:
                f.write(transcription_text)

            transcriptions[audio_file] = transcription_text
            logging.info(f"Transcription saved to {output_file}")
        except Exception as e:
            logging.error(f"Failed to transcribe {audio_file}: {e}")
   
    return transcriptions

# def extract_transcription(audio_files, transcription_output_folder, chunk_duration=30):
#     transcriptions = {}
#     for audio_file in audio_files:
#         try:
#             # Load the audio file
#             audio = AudioSegment.from_file(audio_file)
#             total_duration = len(audio) / 1000  # Convert duration to seconds
            
#             transcription_text = ""
#             for i in range(0, int(total_duration), chunk_duration):
#                 # Extract a chunk of the audio
#                 start_time = i * 1000
#                 end_time = min((i + chunk_duration) * 1000, len(audio))
#                 audio_chunk = audio[start_time:end_time]

#                 # Save the chunk to a temporary file
#                 chunk_path = f"{audio_file}_chunk_{i}.wav"
#                 audio_chunk.export(chunk_path, format="wav")

#                 # Transcribe the chunk
#                 transcription_result = transcription_pipeline(chunk_path)
#                 transcription_text += transcription_result.get('text', '') + " "

#                 # Remove the temporary file
#                 os.remove(chunk_path)

#             # Save the full transcription to the output folder
#             output_file = os.path.join(transcription_output_folder, f"{os.path.basename(audio_file)}.txt")
#             with open(output_file, "w") as f:
#                 f.write(transcription_text.strip())

#             transcriptions[audio_file] = transcription_text.strip()
#             logging.info(f"Transcription saved to {output_file}")
#         except Exception as e:
#             logging.error(f"Failed to transcribe {audio_file}: {e}")

#     return transcriptions


def process_audio_file(audio_file):
    logging.info(f"Running diarization on {audio_file}...")
    diarization_result = diarization_pipeline(audio_file)
    upload_urls, diarized_files = extract_and_upload_speaker_segments(audio_file, diarization_result)
    return upload_urls, diarized_files

def process_transcription(diarized_files):
    logging.info(f"Processing transcription for diarized files...")
    transcriptions = extract_transcription(diarized_files, OUTPUT_TRANSCRIPTION_FOLDER)
    return transcriptions
