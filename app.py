from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn.functional as F
import clip

UPLOAD_FOLDER = 'uploads'
CROP_FOLDER = 'cropped'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)

clip_model, preprocess_clip = clip.load("ViT-B/32", device="cpu")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def crop_and_save(image_path, x, y, w, h, part, name):
    img = Image.open(image_path)
    cropped = img.crop((x, y, x + w, y + h))
    part_path = os.path.join(CROP_FOLDER, f"{part}_{name}.png")
    cropped.save(part_path)


def remove_alpha(image: Image.Image, background=(255, 255, 255)):
    if image.mode in ('RGBA', 'LA'):
        bg = Image.new("RGB", image.size, background)
        bg.paste(image, mask=image.split()[-1])
        return bg
    else:
        return image.convert("RGB")


def get_image_embedding(image: Image.Image):
    image = remove_alpha(image)
    image_input = preprocess_clip(image).unsqueeze(0).to("cpu")
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().squeeze(0)


def find_similar(part: str, image: Image.Image, top_k=1):
    embedding = get_image_embedding(image).unsqueeze(0)
    data = torch.load(f"{part}_embeddings.pt", weights_only=True)
    embeddings = data["embeddings"].cpu()
    filenames = data["filenames"]
    sim = F.cosine_similarity(embedding, embeddings)
    top_indices = torch.topk(sim, k=top_k).indices.tolist()
    return [filenames[i] for i in top_indices]


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('crop_parts', filename=filename, step='head'))
    return render_template('upload.html')


@app.route('/crop/<filename>', methods=['GET', 'POST'])
def crop_parts(filename):
    step = request.args.get('step', 'head')
    if request.method == 'POST':
        x = int(float(request.form['x']))
        y = int(float(request.form['y']))
        width = int(float(request.form['width']))
        height = int(float(request.form['height']))

        name = os.path.splitext(filename)[0]
        crop_and_save(os.path.join(UPLOAD_FOLDER, filename), x, y, width, height, step, name)

        next_step = {'head': 'torso', 'torso': 'legs', 'legs': 'results'}[step]
        if next_step == 'results':
            return redirect(url_for('show_results', name=name))
        return redirect(url_for('crop_parts', filename=filename, step=next_step))

    title_map = {'head': 'Step 1: Crop the HEAD', 'torso': 'Step 2: Crop the TORSO', 'legs': 'Step 3: Crop the LEGS'}
    return render_template('crop.html', filename=filename, title=title_map.get(step, 'Crop'), step=step)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/cropped/<filename>')
def cropped_file(filename):
    return send_from_directory(CROP_FOLDER, filename)


@app.route('/results/<name>')
def show_results(name):
    head_path = os.path.join(CROP_FOLDER, f"head_{name}.png")
    torso_path = os.path.join(CROP_FOLDER, f"torso_{name}.png")
    legs_path = os.path.join(CROP_FOLDER, f"legs_{name}.png")

    try:
        head_match = find_similar("head", Image.open(head_path))[0]
        torso_match = find_similar("torso", Image.open(torso_path))[0]
        legs_match = find_similar("legs", Image.open(legs_path))[0]

        head_id = head_match[5:-4]
        torso_id = torso_match[6:-4]
        legs_id = legs_match[5:-4]

        head_url = f"https://img.bricklink.com/ItemImage/MN/0/{head_id}.png"
        torso_url = f"https://img.bricklink.com/ItemImage/MN/0/{torso_id}.png"
        legs_url = f"https://img.bricklink.com/ItemImage/MN/0/{legs_id}.png"

    except Exception as e:
        return f"❌ Error during similarity search: {e}"

    return render_template('results.html', name=name,
                           head=head_match, torso=torso_match, legs=legs_match,
                           head_url=head_url, torso_url=torso_url, legs_url=legs_url)


if __name__ == '__main__':
    app.run(debug=True)
