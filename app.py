from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
with open('credit_card_predict.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Preprocess data here
    prediction = model.predict([data])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
