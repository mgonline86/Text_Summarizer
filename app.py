from flask import Flask, render_template, request, jsonify, abort, send_file
from summarize import summarize_text
from word2vec_v_2 import main

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods = ['POST'])
def convert_text():
    try:
        body = request.get_json()
        original_text = str(body['text'])
        percent = 0.2
        original_text = body['text']
        converted_data = summarize_text(original_text, percent)
        main(original_text)
        data = {
            'success' : True,
            'original_text' : original_text,
            'summary' : converted_data['summary'],
            'higlighted_summary' : converted_data['higlighted_summary'],
        }
        return jsonify(data)
    except:
        abort(422)

@app.route('/get_image_1')
def get_image_1():
    filename = 'static/img/top_2_bottom_map.png'
    return send_file(filename, mimetype='image/png')

@app.route('/get_image_2')
def get_image_2():
    filename = 'static/img/central_map.png'
    return send_file(filename, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)