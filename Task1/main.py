import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


data_file = os.path.join("archive (3)", "Genre Classification Dataset", "train_data.txt")

data = []
with open(data_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
print(f"Total lines read: {len(lines)}")

for line in lines:
    parts = line.strip().split("::")
    if len(parts) == 4:
        id, title, genre, plot = parts
        if plot.strip():
            data.append((id.strip(), title.strip(), genre.strip(), plot.strip()))

df = pd.DataFrame(data, columns=["id", "title", "genre", "plot"])
print("Loaded valid rows:", df.shape)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_plot"] = df["plot"].apply(clean_text)
df = df[df["clean_plot"].str.strip().astype(bool)]
print("Number of samples after cleaning:", len(df))

if not df.empty:
    print("Sample cleaned plot:", df["clean_plot"].iloc[0][:200])
else:
    print("No data left after cleaning. Please check your file format.")
    exit()


tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X = tfidf.fit_transform(df["clean_plot"])
y = df["genre"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))  


print("\n🎬 Sample Predictions (10 examples):")
sample_indices = y_test.index[:10]

for idx in sample_indices:
    original_plot = df.loc[idx, 'plot']
    actual_genre = df.loc[idx, 'genre']
    transformed = tfidf.transform([clean_text(original_plot)])
    predicted_genre = model.predict(transformed)[0]

    print("\n📝 Plot Summary:")
    print(original_plot[:300] + "...")  
    print(f"✅ Actual Genre   : {actual_genre}")
    print(f"🤖 Predicted Genre: {predicted_genre}")