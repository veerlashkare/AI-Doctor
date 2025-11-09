# 🧠 Cancer Prediction Project

## 🔧 Setup

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


## 📊 Train Model
python -c “from utilities import load_tabular, train_and_save; df=load_tabular(‘data.csv’); train_and_save(df)”

## 🌐 Run Flask API

python app.py

## 💻 Run Streamlit
streamlit run streamlit_app.py
