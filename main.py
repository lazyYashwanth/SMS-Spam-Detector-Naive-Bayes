import pandas as pd
import zipfile
import os
# Import your custom logic
from preprocess import clean_text
from model import NaiveBayesDetector
# Import Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def main():
    zip_path = 'archive.zip'

    if not os.path.exists(zip_path):
        print(f"❌ Error: {zip_path} not found in the project folder.")
        return

    try:
        # 1. LOAD DATA FROM ZIP
        print("📂 Opening archive.zip...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_name = [f for f in z.namelist() if f.endswith('.csv')][0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, encoding='latin-1')

        # Cleanup
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
        df.columns = ['label', 'text']
        df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

        # 2. PREPROCESS
        print("🧹 Cleaning all messages (Text to Tokens)...")
        df['clean_text'] = df['text'].apply(clean_text)

        # 3. SPLIT (80% Training, 20% Testing)
        # We shuffle the data to ensure the model learns properly
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split = int(len(df) * 0.8)
        train_df = df[:split]

        # 4. TRAIN THE MODEL
        print("🧠 Training Naive Bayes model...")
        nb = NaiveBayesDetector()
        nb.fit(train_df['clean_text'], train_df['label_num'])

        # 5. CLASSIFY EVERY MESSAGE (Including those it hasn't seen)
        print("🏷️ Classifying entire dataset...")
        df['model_prediction'] = df['clean_text'].apply(lambda x: nb.predict(x))
        df['prediction_label'] = df['model_prediction'].map({1: 'SPAM', 0: 'HAM'})

        # 6. CALCULATE SCIENTIFIC METRICS
        y_true = df['label_num']
        y_pred = df['model_prediction']

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)

        # 7. PRINT PERFORMANCE REPORT
        print("\n" + "=" * 40)
        print("📊 SMS SPAM DETECTOR PERFORMANCE")
        print("=" * 40)
        print(f"✅ Accuracy:  {acc * 100:.2f}%")
        print(f"🎯 Precision: {prec * 100:.2f}%")
        print(f"📞 Recall:    {rec * 100:.2f}%")
        print(f"🧪 F1 Score:   {f1 * 100:.2f}%")
        print("-" * 40)
        print("🧱 CONFUSION MATRIX:")
        print(f"Actual HAM identified as HAM: {conf_matrix[0][0]}")
        print(f"Actual HAM identified as SPAM (Error): {conf_matrix[0][1]}")
        print(f"Actual SPAM identified as HAM (Missed): {conf_matrix[1][0]}")
        print(f"Actual SPAM identified as SPAM: {conf_matrix[1][1]}")
        print("=" * 40)

        # 8. SAVE RESULTS TO CSV
        output_file = 'classified_results.csv'
        df[['text', 'label', 'prediction_label']].to_csv(output_file, index=False)
        print(f"\n💾 Every message has been labeled and saved to: {output_file}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()