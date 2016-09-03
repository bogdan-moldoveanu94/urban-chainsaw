import os
from flask import Flask, render_template, request, send_from_directory, url_for

UPLOAD_FOLDER = '/src/FFTApp/uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

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