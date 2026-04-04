# RTW Prediction Model

## 📌 Project Goal and Background

This project is a basic machine learning model built to predict Return-to-Work (RTW) outcomes using workers compensation claims data.

The idea behind this project is to understand how real-world claim data can be used to estimate when or how a person might return to work after an incident.

---

## 📊 Dataset Description

The dataset contains records of workers compensation claims with different details about the incident, the person, and the claim.

### Columns:

* **Claim No** → Unique ID for each claim
* **Occurrence No** → Represents a specific occurrence within a claim
* **Claim Financial Year** → Financial year in which the claim was recorded
* **Agency NN** → Agency related to the claim
* **Incident Date** → Date when the incident happened
* **Finalised Date** → Date when the claim process was completed
* **Total Paid** → Total amount paid for the claim
* **Paid Days Lost** → Number of days lost due to injury
* **Service Start Date** → Start date of treatment/service
* **Service End Date** → End date of treatment/service
* **Injury Agency Group** → Group category of the injury agency
* **Bodily Location Group** → Grouped body location of injury
* **Mechanism Group** → Grouped cause of injury
* **Nature Group** → Grouped type of injury
* **Bodily Location** → Exact body part affected
* **Mechanism** → How the injury happened
* **Nature** → Type of injury
* **Major** → Indicates major classification
* **Occupation** → Job role of the person
* **Gender** → Gender of the person
* **Date of Birth** → Birth date of the person
* **Days to RTW** → Number of days taken to return to work
* **RTW Category** → Category of return-to-work outcome
* **Age at Accident Date** → Age at the time of incident
* **Settled** → Whether the claim was settled or not

---

## 🗂️ Project Folder Structure

```
rtw-prediction-model/
├── data/
│   └── sample/
├── docs/
├── notebooks/
├── src/
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── tests/
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt

---

## ⚙️ How to Install and Run Locally

```bash
git clone https://github.com/japinderofficial-hub/rtw-prediction-model.git
cd rtw-prediction-model
pip install -r requirements.txt
```

---

## 🤝 How to Contribute

You can check the contribution guidelines here:
https://github.com/AI-Research-Innovation-Society/rtw-prediction-model/blob/develop/CONTRIBUTING.md

