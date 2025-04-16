📘 `README.md`

```markdown
# 🧠 End-to-End Machine Learning Pipeline

This project demonstrates an end-to-end MLOps workflow for a heart disease prediction model using a structured ML pipeline.
It covers everything from data preprocessing and model training to API deployment and version control — all production-ready and scalable!

---

# 🚀 Project Structure


.
├── data/                      # Raw & processed data
├── notebooks/                # Jupyter notebooks for EDA & testing
├── src/                      # Source code
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── config.yaml
├── model/                    # Saved models
├── app/                      # FastAPI app
│   ├── main.py
│   └── request_example.json
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 📊 Problem Statement

Predict whether a patient is likely to have **Heart disease** based on health and demographic features. The model is trained on a CSV dataset (`heart_disease.csv`) and served via an API.

---

## 🔧 Tech Stack

- **Python 3.12**
- **scikit-learn**, **pandas**, **numpy**
- **FastAPI** for REST API
- **Uvicorn** for ASGI server
- **Docker** for containerization
- **Git & GitHub** for version control

---

## 🛠️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/vigneshwarmr/End-To-End-Machine-Learning-Pipeline.git
cd End-To-End-Machine-Learning-Pipeline
```

2. **Create virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Run training**

```bash
python src/train.py
```

4. **Start the API**

```bash
uvicorn app.main:app --reload
```

5. **Test the API**

```bash
curl -X POST "http://127.0.0.1:8080/predict" \
-H "Content-Type: application/json" \
-d '{"features": [[1, 45, 2, 1, 20, 0, 0, 1, 0, 240, 0, 1, 2, 3, 4]]}'
```

---

## 📦 Docker (Optional)

To build and run using Docker:

```bash
docker build -t heart-disease-api .
docker run -p 8080:8080 heart-disease-api
```

---

## ✅ Features

- Modular and clean codebase
- Model serialization with `joblib`
- FastAPI serving and testing
- Easily extendable with CI/CD and monitoring

---

## 📌 TODOs

- [ ] Add MLflow or DVC for experiment tracking
- [ ] Integrate GitHub Actions for CI/CD
- [ ] Add frontend UI (optional)

---

## 🧑‍💻 Author

**M R Vigneshwar**  
**G Harshavardhini**
[GitHub](https://github.com/vigneshwarmr) || [GitHub](https://github.com/godina-harshavardhini)
[LinkedIn](https://www.linkedin.com/in/vigneshwarmr)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).



---

