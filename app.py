from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib
app = Flask(__name__)

# Đảm bảo rằng mô hình được lưu đúng cách
try:
    with open('./models/random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
except EOFError:
    print("EOFError: Tệp mô hình bị lỗi hoặc rỗng")
    model = None
except Exception as e:
    print(f"Lỗi khác: {e}")
    model = None
model = joblib.load('./models/random_forest_model.pkl');
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    if model is None:
        return 'Model không được tải. Vui lòng kiểm tra lại tệp mô hình.'

    try:
        data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        data = np.array(data).reshape(1, -1)
        prediction = model.predict(data)
        print(f'{prediction}')
        diagnosis = "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"

        return f'Diagnosis: {diagnosis}'
    except Exception as e:
        return f'Error occurred: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask, render_template, request
# import pickle
# import numpy as np
# import joblib

# app = Flask(__name__)

# # Đảm bảo rằng mô hình được lưu đúng cách
# try:
#     with open('./models/random_forest_model.pkl', 'rb') as file:
#         model = pickle.load(file)
# except EOFError:
#     print("EOFError: Tệp mô hình bị lỗi hoặc rỗng")
#     model = None
# except Exception as e:
#     print(f"Lỗi khác: {e}")
#     model = None

# model = joblib.load('./models/random_forest_model.pkl')

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         if model is None:
#             return 'Model không được tải. Vui lòng kiểm tra lại tệp mô hình.'

#         try:
#             data = [
#                 float(request.form['pregnancies']),
#                 float(request.form['glucose']),
#                 float(request.form['blood_pressure']),
#                 float(request.form['skin_thickness']),
#                 float(request.form['insulin']),
#                 float(request.form['bmi']),
#                 float(request.form['dpf']),
#                 float(request.form['age'])
#             ]

#             data = np.array(data).reshape(1, -1)
#             prediction = model.predict(data)
#             print(f'{prediction}')

#             if prediction[0] == 1:
#                 diagnosis = "Positive for Diabetes"
#                 warning = "Bạn có nguy cơ cao, vui lòng đi kiểm tra sức khỏe."
#             else:
#                 diagnosis = "Negative for Diabetes"
#                 warning = "Sức khỏe tốt"

#             return render_template('index.html', diagnosis=diagnosis, warning=warning)
#         except Exception as e:
#             return f'Error occurred: {str(e)}'
#     else:
#         return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)
