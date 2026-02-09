# 📩 SMS Spam Detector (Naive Bayes from Scratch)

A high-performance SMS Spam classifier built with Python. This project implements the **Multinomial Naive Bayes** algorithm from the ground up to classify messages as either "Ham" (normal) or "Spam".

## 📊 Performance Results
The model was trained on the UCI SMS Spam Collection dataset and achieved the following results:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98.76%** |
| **Precision** | **95.08%** |
| **Recall** | **95.72%** |
| **F1-Score** | **95.40%** |

### 🧱 Confusion Matrix
- **True Ham:** 4788 | **False Spam (Error):** 37
- **False Ham (Missed):** 32 | **True Spam:** 715

## 🧠 How it Works
1. **Preprocessing:** Text is lowercased, punctuation is removed, and sentences are tokenized.
2. **Probability Logic:** Uses **Bayes' Theorem** to calculate the likelihood of a message being spam based on word frequency.
3. **Laplace Smoothing:** Implemented ($+1$ smoothing) to handle words that don't appear in the training set.
4. **Log-Likelihood:** Uses log-sum probabilities to prevent numerical underflow and ensure calculation stability.

## 🛠️ Tech Stack
- **Python 3.x**
- **Pandas/NumPy** (Data Handling)
- **Scikit-Learn** (Evaluation Metrics only)

## 🚀 How to Run
1. Clone the repo: `git clone https://github.com/lazyYashwanth/SMS-Spam-Detector-Naive-Bayes.git`
2. Ensure you have `archive.zip` in the root folder.
3. Run the controller:
   ```bash
   python main.py