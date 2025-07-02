from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # print("DATA DITERIMA:", data)  # Debugging
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id wajib disertakan'}), 400

    model_dir = f'models/user_{user_id}'
    try:
        model = joblib.load(os.path.join(model_dir, 'model_kelulusan.pkl'))
        le_bekerja = joblib.load(os.path.join(model_dir, 'le_bekerja.pkl'))
        le_menikah = joblib.load(os.path.join(model_dir, 'le_menikah.pkl'))
        le_status = joblib.load(os.path.join(model_dir, 'le_status.pkl'))
    except Exception as e:
        return jsonify({'error': 'Model belum tersedia. Lakukan retrain terlebih dahulu.', 'details': str(e)}), 500

    try:
        input_data = pd.DataFrame([{
            'STATUS BEKERJA': data['status_bekerja'].strip().upper(),
            'UMUR': float(data['umur']),
            'STATUS MENIKAH': data['status_menikah'].strip().upper(),
            'IPS 1': float(data['ips_1']),
            'IPS 2': float(data['ips_2']),
            'IPS 3': float(data['ips_3']),
            'IPS 4': float(data['ips_4']),
        }])

        input_data['IPK'] = input_data[['IPS 1', 'IPS 2', 'IPS 3', 'IPS 4']].mean(axis=1)

        input_data['STATUS BEKERJA'] = le_bekerja.transform(input_data['STATUS BEKERJA'])
        input_data['STATUS MENIKAH'] = le_menikah.transform(input_data['STATUS MENIKAH'])

        X_pred = input_data[['STATUS BEKERJA', 'UMUR', 'STATUS MENIKAH', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPK']]
        prediction = model.predict(X_pred)[0]
        prediction_label = le_status.inverse_transform([prediction])[0]

        return jsonify({'prediction': prediction_label})
    except Exception as e:
        return jsonify({'error': 'Gagal melakukan prediksi', 'details': str(e)}), 400

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    data = request.json
    user_id = data.get('user_id')
    records = data.get('records', [])

    if not user_id or not records:
        return jsonify({'error': 'user_id dan records wajib disertakan'}), 400

    model_dir = f'models/user_{user_id}'
    try:
        model = joblib.load(os.path.join(model_dir, 'model_kelulusan.pkl'))
        le_bekerja = joblib.load(os.path.join(model_dir, 'le_bekerja.pkl'))
        le_menikah = joblib.load(os.path.join(model_dir, 'le_menikah.pkl'))
        le_status = joblib.load(os.path.join(model_dir, 'le_status.pkl'))
    except Exception as e:
        return jsonify({'error': 'Model belum tersedia.', 'details': str(e)}), 500

    try:
        df = pd.DataFrame(records)
        df['IPK'] = df[['IPS 1', 'IPS 2', 'IPS 3', 'IPS 4']].mean(axis=1)
        df['STATUS BEKERJA'] = le_bekerja.transform(df['STATUS BEKERJA'].str.strip().str.upper())
        df['STATUS MENIKAH'] = le_menikah.transform(df['STATUS MENIKAH'].str.strip().str.upper())

        X_pred = df[['STATUS BEKERJA', 'UMUR', 'STATUS MENIKAH', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPK']]
        predictions = model.predict(X_pred)
        predictions = le_status.inverse_transform(predictions)

        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': 'Gagal memproses batch', 'details': str(e)}), 400


@app.route('/retrain', methods=['POST'])
def retrain_model():
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id tidak ditemukan'}), 400

    file = request.files['dataset']
    if not file or not file.filename.endswith('.xlsx'):
        return jsonify({'error': 'File tidak valid'}), 400

    try:
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip().str.upper()
        df = df.dropna()

        required_columns = ['STATUS BEKERJA', 'UMUR', 'STATUS MENIKAH', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'STATUS KELULUSAN']
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': 'Kolom tidak lengkap'}), 400

        df['IPK'] = df[['IPS 1', 'IPS 2', 'IPS 3', 'IPS 4']].mean(axis=1)

        le_bekerja = LabelEncoder()
        le_menikah = LabelEncoder()
        le_status = LabelEncoder()

        df['STATUS BEKERJA'] = le_bekerja.fit_transform(df['STATUS BEKERJA'].astype(str).str.strip().str.upper())
        df['STATUS MENIKAH'] = le_menikah.fit_transform(df['STATUS MENIKAH'].astype(str).str.strip().str.upper())
        df['STATUS KELULUSAN'] = le_status.fit_transform(df['STATUS KELULUSAN'].astype(str).str.strip().str.upper())

        X = df[['STATUS BEKERJA', 'UMUR', 'STATUS MENIKAH', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4', 'IPK']]
        y = df['STATUS KELULUSAN']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = GaussianNB()
        model.fit(X_train, y_train)

        model_dir = f'models/user_{user_id}'
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(model, os.path.join(model_dir, 'model_kelulusan.pkl'))
        joblib.dump(le_bekerja, os.path.join(model_dir, 'le_bekerja.pkl'))
        joblib.dump(le_menikah, os.path.join(model_dir, 'le_menikah.pkl'))
        joblib.dump(le_status, os.path.join(model_dir, 'le_status.pkl'))

        return jsonify({'message': 'Model berhasil dilatih ulang'})
    except Exception as e:
        return jsonify({'error': 'Gagal melakukan pelatihan ulang', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
