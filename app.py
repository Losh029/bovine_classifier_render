import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from utils import load_breed_data, predict, decode_base64_image
from PIL import Image
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['ALLOWED_EXTENSIONS'] = {'png','jpg','jpeg'}

# Load breed CSV
breed_data = load_breed_data()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        lang = request.form.get("language")
        return redirect(url_for('upload', lang=lang))
    return render_template("index.html")

@app.route("/upload/<lang>", methods=["GET", "POST"])
def upload(lang):
    if request.method == "POST":
        # File upload
        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return "No selected file"
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                result = predict(filepath, breed_data, lang)
                return render_template("result.html", result=result, img_filename=filename)
        # Camera capture
        elif "camera_image" in request.form:
            img = decode_base64_image(request.form["camera_image"])
            filename = "camera_capture.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(filepath)
            result = predict(img, breed_data, lang)
            return render_template("result.html", result=result, img_filename=filename)
    return render_template("upload.html", lang=lang)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
