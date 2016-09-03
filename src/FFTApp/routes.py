import os
from flask import Flask, render_template, request, send_from_directory, url_for

UPLOAD_FOLDER = '/src/FFTApp/uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/about')
def about():
        return render_template('about.html')

@app.route('/show/<filename>')
def uploaded_file(filename):
    # filename = 'https://urban-chainsaw.herokuapp.com/uploads/' + filename
    return render_template('imageTemplate.html', filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/start', methods=['POST'])
def get_counts():
    # get url
    data = json.loads(request.data.decode())
    url = data["url"]
    if 'http://' not in url[:7]:
        url = 'http://' + url
    # start job
    job = q.enqueue_call(
        func=count_and_save_words, args=(url,), result_ttl=5000
    )
    # return created job id
    return job.get_id()