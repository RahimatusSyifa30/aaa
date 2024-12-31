from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import pickle

app = Flask(__name__)

app.secret_key = 'your_secret_key'

# Fungsi untuk melatih dan mengevaluasi model SVM menggunakan K-Fold Cross Validation
def train_and_evaluate_model(gamma, lambd, C, epsilon, data_path, k=5):
    # Load data
    data = pd.read_csv(data_path)
    labels = data["Label"].map({'positif': 1, 'negatif': -1}).values
    reviews = data["content"].values

    # TF-IDF to numeric representation
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(reviews).toarray()
    y = labels

    # Linear kernel function
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)

    # Initialize parameters alpha, gamma, lambd, C, and epsilon
    def initialize_parameters(n, gamma, lambd, C, epsilon):
        alpha = np.full(n, 0.1)
        return alpha

    # Calculate D_ij matrix
    def calculate_D_matrix(X, y, kernel, lambd):
        n = len(X)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i][j] = y[i] * y[j] * kernel(X[i], X[j]) + lambd ** 2
        return D

    # Update alpha values
    def update_alpha(alpha, D, gamma, epsilon, C, y):
        n = len(alpha)
        max_delta_alpha = 0
        for i in range(n):
            E_i = np.sum(alpha * y * D[i])
            delta_alpha_i = min(max(gamma * (1 - E_i), -alpha[i]), C - alpha[i])
            alpha[i] += delta_alpha_i
            max_delta_alpha = max(max_delta_alpha, abs(delta_alpha_i))
        return alpha, max_delta_alpha < epsilon

    # Calculate w and b values
    def calculate_w_and_b(X, y, alpha, kernel):
        w = np.sum(alpha[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)
        support_vectors_idx = np.where(alpha > 0)[0]
        b = np.mean(y[support_vectors_idx] - np.dot(X[support_vectors_idx], w))
        return w, b

    # Decision function
    def decision_function(X_test, X_train, y_train, alpha, b, kernel):
        h = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            h_i = 0
            for j in range(X_train.shape[0]):
                h_i += alpha[j] * y_train[j] * kernel(X_train[j], X_test[i])
            h[i] = h_i + b
        return h

    # K-Fold Cross Validation function
    def k_fold_cross_validation(X, y, k, gamma, lambd, C, epsilon):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Initialize parameters for this fold
            alpha = initialize_parameters(len(X_train), gamma, lambd, C, epsilon)

            # Calculate D_ij matrix
            D_ij = calculate_D_matrix(X_train, y_train, linear_kernel, lambd)

            # Update alpha values until convergence
            converged = False
            iteration_count = 0
            while not converged:
                alpha, converged = update_alpha(alpha, D_ij, gamma, epsilon, C, y_train)
                iteration_count += 1

            # Calculate w and b values
            w, b = calculate_w_and_b(X_train, y_train, alpha, linear_kernel)

            # Decision function for test data
            h_test = decision_function(X_test, X_train, y_train, alpha, b, linear_kernel)

            # Predict labels for test data
            y_pred = np.sign(h_test)

            # Evaluate using confusion matrix
            TP = np.sum((y_test == 1) & (y_pred == 1))
            TN = np.sum((y_test == -1) & (y_pred == -1))
            FP = np.sum((y_test == -1) & (y_pred == 1))
            FN = np.sum((y_test == 1) & (y_pred == -1))

            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        return {
            'mean_accuracy': np.mean(accuracies),
            'mean_precision': np.mean(precisions),
            'mean_recall': np.mean(recalls),
            'mean_f1': np.mean(f1_scores)
        }

    # Perform K-Fold Cross Validation
    results = k_fold_cross_validation(X_tfidf, y, k, gamma, lambd, C, epsilon)

    positive_percentage = np.sum(labels == 1) / len(labels) * 100
    negative_percentage = np.sum(labels == -1) / len(labels) * 100

    return results['mean_accuracy'], results['mean_precision'], results['mean_recall'], results['mean_f1'], positive_percentage, negative_percentage

@app.route('/', methods=['GET'])
def home():
    datasets = {
        "databersih": "databersih.csv",
        "dataakulaku": "dataakulaku.csv",
        "dataeasycash": "dataeasycash.csv"
    }
    # Default nilai parameter
    gamma, lambd, C, epsilon = 0.5, 0.5, 1.5, 0.001
    selected_dataset = request.args.get('dataset', 'databersih')

    page_titles = ""
    if selected_dataset == "databersih":
        page_titles = "Aplikasi Pinjaman Online AdaKami"
    elif selected_dataset == "dataakulaku":
        page_titles = "Aplikasi Pinjaman Online Akulaku"
    elif selected_dataset == "dataeasycash":
        page_titles = "Aplikasi Pinjaman Online Easycash"

    # Latih dan evaluasi model dengan parameter yang diberikan
    data_path = datasets[selected_dataset]
    mean_accuracy, mean_precision, mean_recall, mean_f1, positive_percentage, negative_percentage = train_and_evaluate_model(gamma, lambd, C, epsilon, data_path)

    positive_percentage = round(positive_percentage, 2)
    negative_percentage = round(negative_percentage, 2)

    return render_template('index.html', mean_accuracy=mean_accuracy, mean_precision=mean_precision, mean_recall=mean_recall, mean_f1=mean_f1, gamma=gamma, lambd=lambd, C=C, epsilon=epsilon, datasets=datasets, selected_dataset=selected_dataset, positive_percentage=positive_percentage, negative_percentage=negative_percentage, page_titles=page_titles)


# @app.route('/admins', methods=['GET', 'POST'])
# def adminss():
# #     return render_template('admin.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True
            return redirect(url_for('admin'))
        else:
            return "Login gagal, coba lagi."
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/admins', methods=['GET', 'POST'])
def admin():
    if 'logged_in' in session:
        datasets = {
            "databersih": "databersih.csv",
            "dataakulaku": "dataakulaku.csv",
            "dataeasycash": "dataeasycash.csv"
        }
        # Default nilai parameter
        gamma, lambd, C, epsilon = 0.5, 0.5, 1.5, 0.001
        selected_dataset = request.args.get('dataset', 'databersih')

        if request.method == 'POST':
            #Ambil parameter dari form input
            gamma = float(request.form['gamma'])
            lambd = float(request.form['lambd'])
            C = float(request.form['C'])
            epsilon = float(request.form['epsilon'])

        page_title = ""
        if selected_dataset == "databersih":
            page_title = "Aplikasi Pinjaman Online AdaKami"
        elif selected_dataset == "dataakulaku":
            page_title = "Aplikasi Pinjaman Online Akulaku"
        elif selected_dataset == "dataeasycash":
            page_title = "Aplikasi Pinjaman Online Easycash"

        # Latih dan evaluasi model dengan parameter yang diberikan
        data_path = datasets[selected_dataset]
        mean_accuracy, mean_precision, mean_recall, mean_f1, positive_percentage, negative_percentage = train_and_evaluate_model(gamma, lambd, C, epsilon, data_path)

        positive_percentage = round(positive_percentage, 2)
        negative_percentage = round(negative_percentage, 2)

        return render_template('admin.html', mean_accuracy=mean_accuracy, mean_precision=mean_precision, mean_recall=mean_recall, mean_f1=mean_f1, gamma=gamma, lambd=lambd, C=C, epsilon=epsilon, datasets=datasets, selected_dataset=selected_dataset, positive_percentage=positive_percentage, negative_percentage=negative_percentage, page_title=page_title)
    else:
        return redirect(url_for('login'))

@app.route('/classify', methods=['POST'])
def classify_review():
    datasets = {
        "databersih": "databersih.csv",
        "dataakulaku": "dataakulaku.csv",
        "dataeasycash": "dataeasycash.csv"
    }
    gamma, lambd, C, epsilon = 0.5, 0.5, 1.5, 0.001
    review = request.form['review']
    selected_dataset = request.form.get('dataset', 'databersih')

    data_path = datasets[selected_dataset]
    mean_accuracy, mean_precision, mean_recall, mean_f1, positive_percentage, negative_percentage = train_and_evaluate_model(gamma, lambd, C, epsilon, data_path)

    data = pd.read_csv(data_path)
    labels = data["Label"].map({'positif': 1, 'negatif': -1}).values
    reviews = data["content"].values

    # TF-IDF to numeric representation
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(reviews).toarray()
    y = labels

    # Linear kernel function
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)

    # Initialize parameters alpha, gamma, lambd, C, and epsilon
    def initialize_parameters(n):
        alpha = np.full(n, 0.1)
        gamma = 0.5  # Learning rate
        lambd = 0.5  # Regularization parameter
        C = 0.5   # Margin parameter
        epsilon = 0.001
        return alpha, gamma, lambd, C, epsilon

    # Calculate D_ij matrix
    def calculate_D_matrix(X, y, kernel, lambd):
        n = len(X)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i][j] = y[i] * y[j] * kernel(X[i], X[j]) + lambd ** 2
        return D

    # Update alpha values
    def update_alpha(alpha, D, gamma, epsilon, C, y):
        n = len(alpha)
        max_delta_alpha = 0
        for i in range(n):
            E_i = np.sum(alpha * y * D[i])
            delta_alpha_i = min(max(gamma * (1 - E_i), -alpha[i]), C - alpha[i])
            alpha[i] += delta_alpha_i
            max_delta_alpha = max(max_delta_alpha, abs(delta_alpha_i))
        return alpha, max_delta_alpha < epsilon

    # Calculate w and b values
    def calculate_w_and_b(X, y, alpha, kernel):
        w = np.sum(alpha[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)
        support_vectors_idx = np.where(alpha > 0)[0]
        b = np.mean(y[support_vectors_idx] - np.dot(X[support_vectors_idx], w))
        return w, b

    # Decision function
    def decision_function(X_test, X_train, y_train, alpha, b, kernel):
        h = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            h_i = 0
            for j in range(X_train.shape[0]):
                h_i += alpha[j] * y_train[j] * kernel(X_train[j], X_test[i])
            h[i] = h_i + b
        return h

    # Use the last trained model values
    alpha, gamma, lambd, C, epsilon = initialize_parameters(len(X_tfidf))
    D_ij = calculate_D_matrix(X_tfidf, y, linear_kernel, lambd)

    converged = False
    iteration_count = 0
    while not converged:
        alpha, converged = update_alpha(alpha, D_ij, gamma, epsilon, C, y)
        iteration_count += 1

    w, b = calculate_w_and_b(X_tfidf, y, alpha, linear_kernel)
    review_tfidf = tfidf_vectorizer.transform([review]).toarray()
    decision_value = np.dot(review_tfidf, w) + b
    prediction = np.sign(decision_value)

    if prediction == 1:
        sentiment = 'positif'
    else:
        sentiment = 'negatif'

    return jsonify({
        'review': review,
        'sentiment': sentiment
    })


datasets = {
    "databersih": "Data Ulasan1.csv",
    "dataakulaku": "Data Ulasan2.csv",
    "dataeasycash": "Data Ulasan3.csv"
}
loaded_data = {key: pd.read_csv(value) for key, value in datasets.items()}

@app.route('/reviews', methods=['GET'])
def get_reviews():
    label = request.args.get('label')
    dataset_key = request.args.get('dataset', 'databersih')
    page = int(request.args.get('page', 1))
    per_page = 5

    dataset = loaded_data.get(dataset_key)
    filtered_reviews = dataset[dataset['Label'] == label]['content']
    total_reviews = len(filtered_reviews)
    start = (page - 1) * per_page
    end = start + per_page

    reviews = filtered_reviews.iloc[start:end].tolist()

    return jsonify({
        'reviews': reviews,
        'total': total_reviews,
        'page': page,
        'per_page': per_page
    })

if __name__ == '__main__':
    app.run(debug=True)
