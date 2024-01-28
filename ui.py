import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog

# Function to extract URL attributes
def extract_attributes(url):
    # Parse the URL
    parsed_url = urlparse(url)

    # Extract components
    url_length = len(url)
    valid_url = 1 if parsed_url.scheme and parsed_url.netloc else 0
    at_symbol = 1 if '@' in parsed_url.netloc else 0
    sensitive_words_count = 0  # You may need to define how to count sensitive words
    path_length = len(parsed_url.path)
    isHttps = 1 if parsed_url.scheme == 'https' else 0
    nb_dots = url.count('.')
    nb_hyphens = url.count('-')
    nb_and = url.count('&')
    nb_or = url.count('|')
    nb_www = 1 if parsed_url.netloc.startswith('www.') else 0
    nb_com = 1 if parsed_url.netloc.endswith('.com') else 0
    nb_underscore = url.count('_')

    # Create a DataFrame with the split attributes
    attributes_df = pd.DataFrame({
        'url_length': [url_length],
        'valid_url': [valid_url],
        'at_symbol': [at_symbol],
        'sensitive_words_count': [sensitive_words_count],
        'path_length': [path_length],
        'isHttps': [isHttps],
        'nb_dots': [nb_dots],
        'nb_hyphens': [nb_hyphens],
        'nb_and': [nb_and],
        'nb_or': [nb_or],
        'nb_www': [nb_www],
        'nb_com': [nb_com],
        'nb_underscore': [nb_underscore]
    })

    return attributes_df

# Function to load and predict based on new data
def load_and_predict():
    # Get URL from user input
    url = url_entry.get()

    # Extract attributes from the URL
    new_data = extract_attributes(url)

    # Standardize the new data and make predictions
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)[0]

    # Display the result in the UI
    results_text.config(state=tk.NORMAL)
    results_text.delete(1.0, tk.END)
    result = 'Fake link detected.' if prediction == 1 else 'Legitimate link detected.'
    results_text.insert(tk.END, f'{url}: {result}\n')
    results_text.config(state=tk.DISABLED)

# Load the dataset
df = pd.read_csv("phishing_url_dataset.csv")

# Extract features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Create a simple UI
root = tk.Tk()
root.title("Phishing Link Detector")

# URL Entry
url_label = tk.Label(root, text="Enter URL:")
url_label.pack(pady=5)
url_entry = tk.Entry(root, width=40)
url_entry.pack(pady=10)

# Load and Predict Button
load_button = tk.Button(root, text="Check Link", command=load_and_predict)
load_button.pack(pady=10)

# Results Text Box
results_text = tk.Text(root, height=3, width=50, state=tk.DISABLED)
results_text.pack(pady=10)

root.mainloop()
