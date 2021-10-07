from flask import Flask, render_template, request, jsonify, abort
from summarize import summarize_text

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
        data = {
            'success' : True,
            'original_text' : original_text,
            'summary' : converted_data['summary'],
            'higlighted_summary' : converted_data['higlighted_summary'],
        }
        return jsonify(data)
    except:
        abort(422)

if __name__ == '__main__':
    app.run()