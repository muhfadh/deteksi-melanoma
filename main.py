import time
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, redirect, render_template
from tensorflow.keras.models import load_model
from pathlib import Path

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

dir1 = ['static/test-images/acral melanoma/', 'static/test-images/benign nevi/']

def get_imglist(path):
    # list_img_path = []
    # for path in input_path: 
    #     for img_path in sorted(Path(path).glob('*.jpg')) or sorted(Path(path).glob('*.jpeg')) or sorted(Path(path).glob('*.png')):
    #         list_img_path.append(img_path)
    # return list_img_path
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
def index():
    list_img_path = []
    for direktori in dir1:
        img_list = get_imglist(direktori)
        for i, img_path in enumerate(img_list):
            img_name = os.path.split(img_path)[1]
            if 'AM' in img_name:
                img_name = os.path.join('static/test-images/acral melanoma/', img_name)
            else:
                img_name = os.path.join('static/test-images/benign nevi/', img_name)
            list_img_path.append(img_name)
    return render_template('/select.html', list_img_path = list_img_path)

# @app.route('/predict-upload', methods=['POST'])
# def predict_upload():
#     chosen_model = request.form['select_model']
#     model_dict = {'hyperModel'   :   'static/MLModule/model-2-89%.h5',
#                   'LRSModel'   :   'static/MLModule/model-6-89%.h5',
#                   'model-v3' : 'static/MLModule/model-v3.h5',
#                   'model-v4': 'static/MLModule/model-v4.h5'}
#     if chosen_model in model_dict:
#         model = load_model(model_dict[chosen_model]) 
#     else:
#         model = load_model(model_dict[0])

#     file = request.files["file"]
#     file.save(os.path.join('static', 'temp.jpg'))
#     img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)
#     img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
#     start = time.time()
#     pred = model.predict(img)[0]
#     labels = (pred > 0.5).astype(np.int)
#     print(labels)
#     runtimes = round(time.time()-start,4)
#     respon_model = [round(elem * 100, 2) for elem in pred]
#     return predict_result(chosen_model, runtimes, respon_model, 'temp.jpg', labels, pred)


# @app.route('/predict-sample', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         filename = request.form.get('input_image')

#         chosen_model = request.form['select_model_sample']
#         model_dict = {'hyperModel'   :   'static/MLModule/model-2-89%.h5',
#                     'LRSModel'   :   'static/MLModule/model-6-89%.h5',
#                     'model-v3' : 'static/MLModule/model-v3.h5',
#                     'model-v4': 'static/MLModule/model-v4.h5'}
#         if chosen_model in model_dict:
#             model = load_model(model_dict[chosen_model]) 
#         else:
#             model = load_model(model_dict[0])

#         img = Image.open(filename)
#         img.save(os.path.join('static', 'temp.jpg'))
#         img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
#         img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
#         start = time.time()
#         pred = model.predict(img)[0]
#         labels = (pred > 0.5).astype(np.int)
#         print(labels)
#         runtimes = round(time.time()-start,4)
#         respon_model = [round(elem * 100, 2) for elem in pred]
#         return predict_result(chosen_model, runtimes, respon_model, 'temp.jpg', labels, pred)

@app.route('/predict-upload-compare', methods=['POST'])
def predict_upload_compare():
    chosen_model = request.form.getlist('check_model')
    model_dict = {'hyperModel'   :   'static/MLModule/model-2-89%.h5',
                  'LRSModel'   :   'static/MLModule/model-6-89%.h5',
                  'model-v3' : 'static/MLModule/model-v3.h5',
                  'model-v4': 'static/MLModule/model-v4.h5',
                  'vgg19': 'static/MLModule/vgg-model.h5'}
    file = request.files["file"]
    file.save(os.path.join('static', 'temp.jpg'))
    runtimes_list = []
    respon_model_list = []
    labels_list = []
    pred_list = []
    for m in chosen_model:
        if m in model_dict:
            model = load_model(model_dict[m]) 
        else:
            model = load_model(model_dict['hyperModel'])
            
        img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)
        img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
        start = time.time()
        pred = model.predict(img)[0]
        labels = (pred > 0.5).astype(np.int)
        runtimes = round(time.time()-start,4)
        respon_model = [round(elem, 2) for elem in pred]

        runtimes_list.append(runtimes)
        respon_model_list.append(respon_model)
        labels_list.append(labels)
        pred_list.append(pred)

    return predict_result_compare(chosen_model, runtimes_list, respon_model_list, 'temp.jpg', labels_list, pred_list)

@app.route('/predict-sample-compare', methods=['POST'])
def predict_sample_compare():
    if request.method == 'POST':

        runtimes_list = []
        respon_model_list = []
        labels_list = []
        pred_list = []
        filename = request.form.get('input_image')
        img = Image.open(filename)
        img.save(os.path.join('static', 'temp.jpg'))
        chosen_model = request.form.getlist('check_model_sample')
        model_dict = {'hyperModel'   :   'static/MLModule/model-2-89%.h5',
                  'LRSModel'   :   'static/MLModule/model-6-89%.h5',
                  'model-v3' : 'static/MLModule/model-v3.h5',
                  'model-v4': 'static/MLModule/model-v4.h5',
                  'vgg19': 'static/MLModule/vgg-model.h5'}
        for m in chosen_model:
            if m in model_dict:
                model = load_model(model_dict[m]) 
            else:
                model = load_model(model_dict['hyperModel'])

            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
            start = time.time()
            pred = model.predict(img)[0]
            labels = (pred > 0.5).astype(np.int)
            runtimes = round(time.time()-start,4)
            respon_model = [round(elem, 2) for elem in pred]

            runtimes_list.append(runtimes)
            respon_model_list.append(respon_model)
            labels_list.append(labels)
            pred_list.append(pred)

    return predict_result_compare(chosen_model, runtimes_list, respon_model_list, 'temp.jpg', labels_list, pred_list)

# def predict_result(model, run_time, probs, img, labels, pred):
#     class_list = ['acral melanoma', 'benign nevi']
#     if labels == 0:
#         labels = class_list[0]
#     else:
#         labels = class_list[1]
#     return render_template('/result_select.html', labels=labels, 
#                             probs=probs, model=model,
#                             run_time=run_time, img=img, pred=pred)

def predict_result_compare(model, run_time, probs, img, labels, pred):
    class_list = ['acral melanoma', 'benign nevi']
    list_label = []
    for label in labels:
        if label == 0:
            label = class_list[0]
        else:
            label = class_list[1]
        list_label.append(label)
    return render_template('/result_select_compare.html', len = len(model), labels=list_label, 
                            probs=probs, model=model,
                            run_time=run_time, img=img, pred=pred)

if __name__ == "__main__": 
        app.run(debug=True)