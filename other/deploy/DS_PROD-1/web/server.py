from flask import Flask, request, jsonify
import pickle
import numpy as np

# загружаем модель из файла
with open('models/model.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)

# создаём приложение
app = Flask(__name__)

@app.route('/')
def index():
    msg = "Тестовое сообщение. Сервер запущен!"
    return msg

@app.route('/predict', methods=['POST'])
def predict():
    nums = request.json.get('nums')
    features = np.array(nums).reshape(1, 4)
    res = model.predict(features)
    return jsonify({
        "prediction": res[0]
    })

if __name__ == '__main__':
    app.run('localhost', 5000)