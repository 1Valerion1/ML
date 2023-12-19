from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Загрузка обученной модели
with open('modelTop.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    # Отображение главной страницы с формой для ввода данных
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # получение данных от пользователя
        data = request.form.to_dict()
        data = preprocess(data)
        prediction = model.predict([data])
        return jsonify({'prediction': str(prediction)})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

def preprocess(data):
    required_features = ['instant', 'mnth', 'registered', 'cnt']
    for feature in required_features:
        if feature not in data:
            raise ValueError(f"Отсутствует обязательный признак: {feature}")
    
    # Предобработка данных
    preprocessed_data = np.array([float(data[key]) for key in sorted(data.keys())])
    return preprocessed_data

if __name__ == '__main__':
    app.run(debug=True)
