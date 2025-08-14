from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Welcome to Medical Image Classifier</h1><p>Basic medical image classification demo using TensorFlow</p>"

if __name__ == '__main__':
    app.run(debug=True)
