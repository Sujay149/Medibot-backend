from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from model_logic import transcribe_audio_file, process_query

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# === Health Check ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "MediBot Flask API is running!"})

# === Transcribe Audio ===
@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files["audio"]
    lang = request.form.get("lang", "en")

    try:
        text = transcribe_audio_file(file, lang=lang)
        return jsonify({"transcription": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Process Text Query ===
@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request."}), 400

    user_text = data["text"]
    lang = data.get("lang", "en")

    try:
        response_text, audio_path = process_query(user_text, lang=lang)
        return jsonify({
            "response": response_text,
            "audio_path": f"/api/audio/{audio_path.split('/')[-1]}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Serve Generated Audio ===
@app.route("/api/audio/<filename>")
def get_audio(filename):
    try:
        return send_file(f"audio/{filename}", mimetype="audio/mpeg")
    except FileNotFoundError:
        return jsonify({"error": "Audio file not found."}), 404

# === Run Server ===
if __name__ == "__main__":
    app.run(debug=True, port=5000)
