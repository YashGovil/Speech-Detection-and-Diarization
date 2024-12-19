import os
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from cloud import process_audio_file, process_transcription

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'audio'
app.config['ALLOWED_EXTENSIONS'] = {'wav'}

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Ensure necessary folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("merged_speakers_output", exist_ok=True)
os.makedirs("output_transcription", exist_ok=True)

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to render the HTML frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file uploads and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['audio_file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Run speaker diarization
            upload_urls, diarized_files = process_audio_file(file_path)
            # Return the URLs of uploaded files (e.g., Cloudinary)
            return jsonify({"message": "Processing complete!", "urls": upload_urls}), 200
        except Exception as e:
            logging.error(f"Error during processing: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format. Please upload a WAV file."}), 400

@app.route('/apply_function', methods=['POST'])
def process_action():
    data = request.get_json()
    logging.info(f"Received data: {data}")

    audio_filename = data.get('audio_filename')
    action = data.get('action')

    if not audio_filename or not action:
        logging.error("Missing audio_filename or action")
        return jsonify({"error": "Missing audio filename or action."}), 400

    try:
        diarized_file_folder = "merged_speakers_output"
        file_path = os.path.join(diarized_file_folder, audio_filename)

        if not os.path.exists(file_path):
            logging.error(f"File {audio_filename} not found at {file_path}.")
            # Attempt to match filenames case-insensitively
            potential_files = [f for f in os.listdir(diarized_file_folder) if audio_filename.lower() in f.lower()]
            if potential_files:
                file_path = os.path.join(diarized_file_folder, potential_files[0])  # Use the first match
            else:
                return jsonify({"error": f"File {audio_filename} does not exist."}), 404

        if action == "transcription":
            # Call the transcription function
            transcriptions = process_transcription([file_path])
            transcription_text = list(transcriptions.values())[0]  # Get the transcription text
            return jsonify({"message": "Transcription complete!", "transcription": transcription_text}), 200
        else:
            logging.error(f"Unsupported action: {action}")
            return jsonify({"error": "Unsupported action."}), 400
    except Exception as e:
        logging.error(f"Error during function application: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
