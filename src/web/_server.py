import os
from base64 import b64encode
from io import BytesIO
from shutil import rmtree
from src.backends import ImageClassifier
from tempfile import TemporaryDirectory
from patoolib import extract_archive
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
from PIL import Image

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'tmp')
app.config['SECRET_KEY'] = 'PYTORCH is COOL'
ALLOWED_EXTENSIONS = ('png', 'jpg', 'jpeg', 'gif', 'rar', 'zip', 'tz')

with open('resources/data/labels.txt') as labels_file:
    classification_model = ImageClassifier(
        labels=labels_file.read().split('\n'),
        image_size=224,
        normalize_means=[0.485, 0.456, 0.406],
        normalize_stds=[0.229, 0.224, 0.225],
        device='cpu'
    )


def allowed_file(filename: str) -> bool:
    return ('.' in filename and
            filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS)


def extract_files(filename: str) -> list[Image.Image]:
    file_extension = filename.split('.')[-1]
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if file_extension in ('png', 'jpg', 'jpeg', 'gif'):
        images = [Image.open(full_filename)]

    else:
        with TemporaryDirectory() as extract_directory:
            extract_archive(full_filename, outdir=extract_directory)
            images = [Image.open(os.path.join(extract_directory, file))
                      for file in os.listdir(extract_directory)
                      if file.split('.')[-1] in ('png', 'jpg', 'jpeg', 'gif')]

    os.remove(full_filename)
    return images


def get_top5(results: dict[str, float]) -> list[str]:
    top5 = [item[0] for item in sorted(results.items(),
                                       key=lambda value: value[1],
                                       reverse=True)[:5]]
    return top5


def get_image_data(image: Image.Image) -> str:
    image_io = BytesIO()
    image.save(image_io, 'PNG')
    dataurl = ('data:image/png;base64,'
               + b64encode(image_io.getvalue()).decode('ascii'))
    return dataurl


def clear_uploads():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Не могу прочитать файл', 400
        file = request.files['file']
        if file.filename == '':
            return 'Нет выбранного файла', 400

        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            return ('Не подходящее расширение файла, допустимые расширения: ' +
                    ', '.join(map(lambda name: '.' + name,
                                  ALLOWED_EXTENSIONS)),
                    400)

        dir_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                str(hash(request.remote_addr)))
        os.mkdir(dir_path, mode=0o777)
        file.save(os.path.join(dir_path, filename))

        return redirect('predict')
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    dir_path = os.path.join(app.config['UPLOAD_FOLDER'],
                            str(hash(request.remote_addr)))
    filename = os.listdir(dir_path)[0]
    images = extract_files(os.path.join(dir_path, filename))
    rmtree(dir_path)
    results = [get_top5(dict_)
               for dict_ in classification_model.predict(images)]
    return render_template('predict_page.html',
                           results=zip(results,
                                       [get_image_data(image)
                                        for image in images]))
