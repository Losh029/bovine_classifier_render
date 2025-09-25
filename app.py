import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from utils import load_model, load_breed_data, predict
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['ALLOWED_EXTENSIONS'] = {'png','jpg','jpeg'}

# Load model and CSV
# Model will be automatically downloaded from Hugging Face if not present locally
model = load_model()
breed_data = load_breed_data()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Landing page - language selection
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        lang = request.form.get("language")
        return redirect(url_for('upload', lang=lang))
    return render_template("index.html")

# Upload / camera capture page
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
                result = predict(filepath, model, breed_data, lang)
                return render_template("result.html", result=result, img_filename=filename)

        # Camera capture
        elif "camera_image" in request.form:
            data = request.form["camera_image"]
            header, encoded = data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            img = Image.open(BytesIO(img_bytes))
            filename = "camera_capture.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(filepath)
            result = predict(img, model, breed_data, lang)
            return render_template("result.html", result=result, img_filename=filename)

    return render_template("upload.html", lang=lang)

if __name__ == "__main__":
    # Use 0.0.0.0 for Render deployment
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
