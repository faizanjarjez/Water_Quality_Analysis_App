# 🌊 Water Quality Analysis Model

This project predicts water potability based on water properties like pH, hardness, solids, chloramines, and more using machine learning models.

---

## 📂 Project Structure
```
Water_Quality_Analysis/
│
├── app.py                  # Flask API to serve the model
├── water_quality_analysis.py  # Model training and evaluation
├── water_quality_model.pkl  # Trained Random Forest model
├── scaler.pkl               # Data normalization scaler
├── templates/
│   └── index.html           # Frontend page
├── static/
│   └── style.css            # Frontend styling
├── water_potability.csv     # Dataset (not included here, add it manually)
└── README.md                # Project documentation
```

---

## 🚀 How to Run

1. **Clone the repository** or download the zip.

2. **Install required libraries:**
   ```bash
   pip install flask pandas scikit-learn matplotlib seaborn
   ```

3. **Train the model:**
   ```bash
   python water_quality_analysis.py
   ```

4. **Start the Flask app:**
   ```bash
   python app.py
   ```

5. **Visit in browser:**  
   [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 📈 Models Used
- Linear Regression
- Random Forest Regressor ✅ (best performing)
- Gradient Boosting Regressor

---

## 📊 Dataset
- Water Potability dataset (available from Kaggle or add your own)

---

## 📌 Features
- Predicts water potability based on input features.
- Compares multiple machine learning models.
- Frontend web form to input values and get predictions.

---

## 🛠️ Future Improvements
- Improve UI design.
- Add deployment on cloud (e.g., Render, Vercel, AWS).
- Add visual graphs of predictions.

---

## 🤝 Contributing
Pull requests are welcome! Feel free to suggest improvements or new features.

---

## 📄 License
This project is open-source and free to use under the MIT License.

---
