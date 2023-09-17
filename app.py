import io
from flask import Flask, render_template, request, jsonify, abort, make_response
from summarize import summarize_text
from word2vec_v_2 import main, main2

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

@app.route('/mindmap-a', methods = ['POST'])
def mind_map_a():
    try:
        body = request.get_json()
        original_text = str(body['text'])
        A_img, B_img = main2(original_text)
        response = make_response(A_img)
        response.headers.set('Content-Type', 'image/png')
        response.headers.set(
            'Content-Disposition', 'inline', filename='a_img.png')
        return response
    except Exception as err:
        print(err)
        abort(422)

@app.route('/mindmap-b', methods = ['POST'])
def mind_map_b():
    try:
        body = request.get_json()
        original_text = str(body['text'])
        A_img, B_img = main2(original_text)
        response = make_response(B_img)
        response.headers.set('Content-Type', 'image/png')
        response.headers.set(
            'Content-Disposition', 'inline', filename='b_img.png')
        return response
    except Exception as err:
        print(err)
        abort(422)

if __name__ == '__main__':
    app.run(host="0.0.0.0")