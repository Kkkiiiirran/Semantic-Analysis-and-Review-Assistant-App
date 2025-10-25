import pickle
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from model_setup import load_model


clf, tokenizer, model = load_model("clf.pkl")


with open("test_data.pkl", "rb") as f:
    X_test, y_test = pickle.load(f)


y_pred = clf.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
