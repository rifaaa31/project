import os
import io
import time
import base64
from datetime import datetime
from typing import Dict, Any

from flask import Flask, request, render_template, jsonify, send_file, Response
from werkzeug.utils import secure_filename

import cv2
import numpy as np
from PIL import Image

import sys

# Ensure local backend modules are importable regardless of cwd
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from predict import predict_from_file, predict_from_ndarray_rgb
from chatbot import generate_response


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "frontend", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

_last_result: Dict[str, Any] = {}
_last_image_path: str = ""


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=_last_result if _last_result else None)


@app.route("/predict", methods=["POST"]) 
def predict_route():
    global _last_result, _last_image_path
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{filename}")
    file.save(save_path)

    try:
        result = predict_from_file(save_path)
        _last_result = {
            **result,
            "image_path": save_path,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        _last_image_path = save_path
        return render_template("index.html", result=_last_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/camera")
def camera_stream():
    def gen():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # Provide a blank frame with message if camera not available
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera not available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', blank)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return

        try:
            desired_fps = 12.0
            frame_interval = 1.0 / desired_fps
            while True:
                start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                result = predict_from_ndarray_rgb(frame)
                text_color = (0, 255, 0)
                label_text = "No Lesion Detected"
                if result.get("is_wrong_image"):
                    text_color = (0, 0, 255)
                    label_text = "Wrong Image — Not a Skin Lesion"
                else:
                    predicted = result.get("class")
                    conf = result.get("confidence")
                    if predicted == "no_lesion":
                        label_text = f"No Lesion Detected ({conf:.2f})"
                        text_color = (200, 200, 0)
                    else:
                        label_text = f"Lesion Detected: {predicted} ({conf:.2f})"
                        text_color = (0, 255, 0)

                        # Draw a central bounding box as a visual focus area
                        h, w = frame.shape[:2]
                        box_w, box_h = int(w * 0.6), int(h * 0.6)
                        x1 = (w - box_w) // 2
                        y1 = (h - box_h) // 2
                        x2 = x1 + box_w
                        y2 = y1 + box_h
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                elapsed = time.time() - start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        finally:
            cap.release()

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/report")
def download_report():
    global _last_result, _last_image_path
    if not _last_result:
        return jsonify({"error": "No prediction available to generate a report."}), 400

    fmt = request.args.get("format", "pdf").lower()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    patient_id = request.args.get("patient_id", f"PAT_{timestamp}")

    if fmt == "csv":
        csv_content = (
            "patient_id,timestamp,class,confidence,severity,is_wrong_image\n"
            f"{patient_id},{_last_result.get('timestamp')},{_last_result.get('class')},{_last_result.get('confidence')},{_last_result.get('severity')},{_last_result.get('is_wrong_image')}\n"
        )
        csv_path = os.path.join(REPORTS_DIR, f"report_{patient_id}_{timestamp}.csv")
        with open(csv_path, "w") as f:
            f.write(csv_content)
        return send_file(csv_path, as_attachment=True, download_name=os.path.basename(csv_path))

    # Default: PDF via fpdf
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Skin Lesion Analysis Report", ln=True, align='C')

    pdf.set_font("Arial", size=11)
    pdf.ln(5)
    pdf.cell(200, 8, txt=f"Patient ID: {patient_id}", ln=True)
    pdf.cell(200, 8, txt=f"Date (UTC): {_last_result.get('timestamp')}", ln=True)
    pdf.cell(200, 8, txt=f"Predicted Class: {_last_result.get('class')}", ln=True)
    pdf.cell(200, 8, txt=f"Confidence: {_last_result.get('confidence')}", ln=True)
    pdf.cell(200, 8, txt=f"Severity: {_last_result.get('severity')}", ln=True)

    # Add image if exists
    if os.path.exists(_last_image_path):
        try:
            pdf.ln(4)
            pdf.cell(200, 8, txt="Image:", ln=True)
            # Fit image width to page margins (approx 180mm), keep aspect
            pdf.image(_last_image_path, x=15, y=None, w=180)
        except Exception:
            pass

    pdf_path = os.path.join(REPORTS_DIR, f"report_{patient_id}_{timestamp}.pdf")
    pdf.output(pdf_path)
    return send_file(pdf_path, as_attachment=True, download_name=os.path.basename(pdf_path))


@app.route("/chatbot", methods=["POST"])
def chatbot_route():
    global _last_result
    data = request.get_json(force=True)
    user_message = data.get("message", "")
    context = _last_result if _last_result else {}
    reply = generate_response(user_message, context)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    # Example run: python backend/app.py
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
