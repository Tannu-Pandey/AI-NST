import os

import torch

from flask import Flask, render_template, send_from_directory

from flask_wtf import FlaskForm

from flask_bootstrap import Bootstrap

from werkzeug.utils import secure_filename

from wtforms import FileField, SubmitField, HiddenField, FloatField

from PIL import Image

from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


app = Flask(__name__)

app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class UploadForm(FlaskForm):

    content = FileField('Content Image')
    style = FileField('Style Image')

    content_path = HiddenField()
    style_path = HiddenField()

    alpha = FloatField('Alpha', default=1.0)

    submit = SubmitField('Transfer Style')


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

encoder = VGGEncoder('vgg_normalised.pth').to(device)

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        'decoder_1.pth',
        map_location=device
    )
)

encoder.eval()
decoder.eval()

print("VGG encoder loaded successfully")
print("Trained decoder loaded successfully")


# --------------------------------------------------
# FILE CHECK
# --------------------------------------------------

def allowed_file(filename):

    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower()
        in app.config['ALLOWED_EXTENSIONS']
    )


# --------------------------------------------------
# STYLE TRANSFER
# --------------------------------------------------

def style_transfer(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha,
    device
):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])

    content_image = transform(content_image)
    style_image = transform(style_image)

    content_image = content_image.unsqueeze(0).to(device)
    style_image = style_image.unsqueeze(0).to(device)

    with torch.no_grad():

        # VGG features
        content_feats = encoder(content_image)
        style_feats = encoder(style_image)

        # AdaIN
        target = adaptive_instance_normalization(
            content_feats[-1],
            style_feats[-1]
        )

        # Alpha blending
        target = (
            alpha * target
            + (1 - alpha) * content_feats[-1]
        )

        # Decoder
        stylized_image = decoder(target)

    return stylized_image


# --------------------------------------------------
# SAVE IMAGE
# --------------------------------------------------

def save_image(image, path):

    image = image.cpu().clone()

    image = image.squeeze(0)

    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)

    image.save(path)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():

        # -----------------------------
        # CONTENT IMAGE
        # -----------------------------

        if form.content.data and form.content.data.filename:

            if allowed_file(form.content.data.filename):

                content_filename = secure_filename(
                    form.content.data.filename
                )

                content_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    content_filename
                )

                form.content.data.save(content_path)

                form.content_path.data = content_filename

            else:
                error = "Invalid content image format."

        else:

            content_filename = form.content_path.data


        # -----------------------------
        # STYLE IMAGE
        # -----------------------------

        if form.style.data and form.style.data.filename:

            if allowed_file(form.style.data.filename):

                style_filename = secure_filename(
                    form.style.data.filename
                )

                style_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    style_filename
                )

                form.style.data.save(style_path)

                form.style_path.data = style_filename

            else:
                error = "Invalid style image format."

        else:

            style_filename = form.style_path.data


        # -----------------------------
        # STYLE TRANSFER
        # -----------------------------

        if content_filename and style_filename:

            content_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                content_filename
            )

            style_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                style_filename
            )

            try:

                content_image = Image.open(
                    content_path
                ).convert('RGB')

                style_image = Image.open(
                    style_path
                ).convert('RGB')

                alpha = float(form.alpha.data)

                # Keep alpha between 0 and 1
                alpha = max(0.0, min(1.0, alpha))

                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device
                )

                result_filename = (
                    'stylized_' + content_filename
                )

                result_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    result_filename
                )

                save_image(
                    stylized_image,
                    result_path
                )

                result_image = result_filename

                print("Style transfer completed!")

            except Exception as e:

                error = str(e)

                print("Error:", error)

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# --------------------------------------------------
# SERVE UPLOADED IMAGES
# --------------------------------------------------

@app.route('/uploads/<filename>')
def send_image(filename):

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )


# --------------------------------------------------
# EXAMPLES
# --------------------------------------------------

@app.route('/examples/<path:filename>')
def send_example(filename):

    return send_from_directory(
        'examples',
        filename
    )


# --------------------------------------------------
# RUN APP
# --------------------------------------------------

if __name__ == '__main__':

    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True
    )