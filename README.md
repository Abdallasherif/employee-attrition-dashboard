# 📊 Employee Attrition Analytics Dashboard

A full end-to-end data analytics project exploring why employees leave — and what HR leaders can do about it. Built on 74,498 synthetic employee records across five industries.

---

## 🔍 What This Project Does

This dashboard transforms raw HR data into actionable insights. Instead of gut feelings, decisions are backed by data — uncovering the real drivers behind employee attrition across demographics, compensation, satisfaction levels, and workplace factors.

---

## 🚀 Live Dashboard

👉 **[Click here to open the dashboard](https://employee-attrition-dashboard-ao.streamlit.app/)**

---

## 📁 Project Structure

```
├── app.py                  ← Streamlit dashboard
├── train.csv               ← Training dataset
├── test.csv                ← Test dataset
├── requirements.txt        ← Dependencies
└── README.md               ← You are here
```

---

## 📌 Key Findings

- 📉 Overall attrition rate sits at **47.5%** across all departments
- 💰 Employees who left earned on average **less** than those who stayed
- 🏠 Remote work shows a notable correlation with employee retention
- 📈 Employees with **zero promotions** leave at significantly higher rates
- ⚖️ Poor work-life balance is one of the strongest predictors of attrition

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python | Core language |
| 🐼 Pandas | Data loading, cleaning & manipulation |
| 📊 Plotly | Interactive visualizations |
| 🎯 Streamlit | Dashboard framework & deployment |
| ☁️ Streamlit Cloud | Hosting |

---

## 📂 Dataset

- **Source:** [Kaggle — Synthetic Employee Attrition Dataset](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset)
- **Size:** 74,498 records across train + test files
- **Features:** 22 columns covering demographics, compensation, satisfaction, and workplace factors
- **Target:** `Attrition` — 0 (stayed) / 1 (left)

> ⚠️ This is a **synthetic dataset**. Patterns are realistic but generated — insights should be treated as analytical exercises, not real-world benchmarks.

---

## ⚙️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOURUSERNAME/employee-attrition-dashboard.git

# 2. Navigate into the folder
cd employee-attrition-dashboard

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📋 Project Pipeline

```
Load & Combine → Clean & Preprocess → EDA → Visualize → Dashboard → Deploy
```

| Step | What Was Done |
|------|--------------|
| 1️⃣ Load & Combine | Merged train and test CSVs into a single 74,498-row dataset |
| 2️⃣ Clean & Preprocess | Handled nulls, encoded ordinal features, normalized column names to snake_case |
| 3️⃣ EDA | Explored attrition patterns across all 22 features |
| 4️⃣ Visualize | Built interactive Plotly charts with clear business takeaways |
| 5️⃣ Dashboard | Shipped a filterable Streamlit app an HR leader can actually use |
| 6️⃣ Deploy | Live on Streamlit Community Cloud |

---

## 🎓 About

This project was built as the **Week 1 Task** of the **Kayfa AI & Data Analytics Internship Program** — Data Analytics Track.

---

*Built with 🔥 by Abdalla Omar*<img width="1212" height="725" alt="image" src="https://github.com/user-attachments/assets/82a6be9f-3e13-4d61-9ecf-a308d0452db6" />
