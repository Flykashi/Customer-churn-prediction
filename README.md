# Client Conversation Risk & Churn Prediction Project

This is my project where I tried to combine **emotion analysis from client conversations** with **customer churn prediction**. 
The idea is to see whether the emotions in conversations can help in predicting which clients might churn or pose a legal risk.

---

# What the Project Does

- Loads two datasets:
  - `test.csv` → contains conversations, emotion labels, and dialogue acts
  - `customer_churn.csv` → contains details like Age, Total Purchases, Account Manager status, Churn labels etc.

- Converts emotion sequences in conversations into a **risk score**  
- Merges this score with regular business data  
- Trains a **Random Forest classifier** to predict churn  
- Also simulates a fake "legal risk" probability  
- Shows results in the form of:
  - Confusion matrix heatmap  
  - Scatter plot (risk vs churn probability)  
  - List of top 10 high-risk clients  

---

# Files Included

| File | Description |
|------|-------------|
| `final.py` | Main script – runs churn + risk prediction + visualizations |
| `beta.py` | Early emotion classifier using TF-IDF + Logistic Regression |
| `test.csv` | Conversation dataset |
| `customer_churn.csv` | Business data for churn prediction |
| `requirements.txt` | All Python libraries used |
| `README.md` | This file (explains what the project is about) |

---

# How to Run

1. Install the required Python libraries using:
   ```bash
   pip install -r requirements.txt

# Output for final.py

Model Accuracy: 0.872

Classification Report:
               precision    recall  f1-score   support

           0       0.88      0.97      0.93       150
           1       0.73      0.37      0.49        30

    accuracy                           0.87       180
   macro avg       0.81      0.67      0.71       180
weighted avg       0.86      0.87      0.85       180


Top 10 Clients at Highest Churn Risk:

         Names                      Company  conversation_risk  legal_risk  Churn_Prob  Account_Manager
 George Archer                    Smith Inc                0.5    0.469920       0.975                1
   Adam Waters                     Reed Ltd                0.5    0.454917       0.970                0
  Craig Garner              Abbott-Thompson                0.5    0.438432       0.965                1
 April Freeman        Larson, Ross and Ward                0.5    0.503888       0.960                1
Joseph Harrell Fitzgerald, Sherman and Lowe                0.5    0.461756       0.960                1
    Tammy Reed                  Russo-Rivas                0.5    0.370861       0.955                1
     Tina King                 Franco-Jones                0.5    0.505305       0.950                1
Jennifer Lynch                Reed-Martinez                0.5    0.666321       0.940                1
   Paul Walker                   Dalton LLC                0.5    0.509128       0.925                0
 Doris Wilkins     Andrews, Adams and Davis                0.5    0.674391       0.925                1

 # output for beta.py

Original dataset shape: (1000, 3)

Expanded dataset shape: (1000, 3)
                                           utterance  act  emotion
0  Hey man , you wanna buy some weed ?  Some what...    3        0
1  The taxi drivers are on strike again .  What f...    1        0
2  We've managed to reduce our energy consumption...    1        0
3  Believe it or not , tea is the most popular be...    1        0
4  What are your personal weaknesses ?  I � m afr...    2        0
c:\Users\Soumyajit\OneDrive\Desktop\customer-churn\beta.py:54: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x="emotion_label", data=df_expanded, palette="viridis")
c:\Users\Soumyajit\OneDrive\Desktop\customer-churn\beta.py:59: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x="act", data=df_expanded, palette="magma")

Classification Report (Emotion Prediction):
C:\Users\Soumyajit\AppData\Local\Programs\Python\Python314\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\Soumyajit\AppData\Local\Programs\Python\Python314\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\Soumyajit\AppData\Local\Programs\Python\Python314\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
              precision    recall  f1-score   support

     disgust       0.00      0.00      0.00         2
        fear       0.00      0.00      0.00         2
   happiness       0.00      0.00      0.00         2
     neutral       0.93      0.93      0.93       178
    surprise       0.38      0.50      0.43        16

    accuracy                           0.87       200
   macro avg       0.26      0.29      0.27       200
weighted avg       0.86      0.87      0.86       200


