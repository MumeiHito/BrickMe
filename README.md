# 🧱 BrickMe

**BrickMe** is a Flask-based web app that lets you upload a full-body photo, crop out your head, torso, and legs, and then matches each part with the most visually similar LEGO minifigure components using CLIP embeddings and BrickLink's image database.


---

## 🚀 How to Run the Project

1. **Clone the repository**

```bash
git clone https://github.com/MumeiHito/BrickMe.git
cd BrickMe
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate.bat    # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Flask app**

```bash
python app.py
```

5. **Open in your browser**

Navigate to:  
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📁 Project Structure

```
.
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── templates/            # HTML pages (upload, crop, results)
├── static/               # Cropper.js and styling
├── uploads/              # Temporary uploaded images
├── cropped/              # Cropped body parts
├── head_embeddings.pt    # Precomputed head embeddings
├── torso_embeddings.pt   # Precomputed torso embeddings
├── legs_embeddings.pt    # Precomputed leg embeddings
```

---

## 🧠 Requirements

- Python 3.9–3.11
- pip
- PyTorch (CPU version is sufficient)