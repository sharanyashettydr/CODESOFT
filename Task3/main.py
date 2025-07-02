import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]  
df.columns = ['label', 'message']


df.dropna(inplace=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}


for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")


model = models['Logistic Regression']
y_pred = model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()


def predict_message(msg, model, vectorizer):
    msg_tfidf = vectorizer.transform([msg])
    prediction = model.predict(msg_tfidf)[0]
    return "Spam" if prediction == 1 else "Ham"


test_msg = "Congratulations! You've won a free iPhone. Click here to claim now."
print("\nTest Message Prediction:")
print("Message:", test_msg)
print("Prediction:", predict_message(test_msg, models["Logistic Regression"], vectorizer))
