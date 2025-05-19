import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ----------- Step 1: Load Dataset -------------
df = pd.read_csv("C:\\Users\\Vaibhav Singh\\OneDrive\\Documents\\spam.csv")
df.columns = ['label', 'text']


# Convert labels to binary (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# ----------- Step 3: Vectorization -------------
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# ----------- Step 4: Train Model -------------
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# ----------- Step 5: Predict & Evaluate -------------
y_pred = model.predict(X_test_vect)
acc = accuracy_score(y_test, y_pred)


print(f"✅ Accuracy: {acc*100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------- Step 6: Save Model & Vectorizer -------------
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("🎉 Model and vectorizer saved successfully!")


# predict_spam.py

import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Input message
message = input("Enter your message: ")

# Vectorize & Predict
message_vect = vectorizer.transform([message])
prediction = model.predict(message_vect)

if prediction[0] == 1:
    print("🚫 Spam message detected.")
else:
    print("✅ This is a normal (ham) message.")