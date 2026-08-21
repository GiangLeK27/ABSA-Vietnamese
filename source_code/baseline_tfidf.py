import os
import numpy as np
import pandas as pd

# Thư viện mô hình và đặc trưng
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Thư viện đánh giá
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# 1. ĐỊNH VỊ VÀ ĐỌC DỮ LIỆU
# ==========================================
# Tự động lấy đường dẫn thư mục hiện tại của file code
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đến thư mục chứa dữ liệu UIT-VSFC 1
DATA_DIR = os.path.join(BASE_DIR, 'data', 'UIT-VSFC 1')

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
dev_df   = pd.read_csv(os.path.join(DATA_DIR, 'dev.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Danh sách các khía cạnh trong bài toán ABSA
aspect_cols = ['LECTURER', 'PROGRAM', 'FACILITY', 'OTHERS']

# Tách Input (X) và Output (y)
X_train, y_train = train_df['Review'].fillna(''), train_df[aspect_cols]
X_dev,   y_dev   = dev_df['Review'].fillna(''),   dev_df[aspect_cols]
X_test,  y_test  = test_df['Review'].fillna(''),  test_df[aspect_cols]

# ==========================================
# 2. TRÍCH XUẤT ĐẶC TRƯNG TỪ VỰNG VỚI TF-IDF
# ==========================================
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_dev_tfidf   = vectorizer.transform(X_dev)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"Kích thước ma trận TF-IDF tập Train: {X_train_tfidf.shape}\n")

# ==========================================
# 3. HÀM ĐÁNH GIÁ VÀ IN KẾT QUẢ CHI TIẾT
# ==========================================
def evaluate_model(model, model_name):
    print(f"=================== MÔ HÌNH: {model_name} ===================")
    
    # Huấn luyện mô hình
    model.fit(X_train_tfidf, y_train)
    
    # Đánh giá trên tập DEV và tập TEST
    for set_name, X_data, y_true in [("TẬP DEV", X_dev_tfidf, y_dev), ("TẬP TEST", X_test_tfidf, y_test)]:
        y_pred = model.predict(X_data)
        acc_list, f1_list = [], []
        
        print(f"\n--- KẾT QUẢ TRÊN {set_name} ---")
        for i, col in enumerate(aspect_cols):
            acc = accuracy_score(y_true.iloc[:, i], y_pred[:, i])
            f1 = f1_score(y_true.iloc[:, i], y_pred[:, i], average='macro')
            acc_list.append(acc)
            f1_list.append(f1)
            print(f"Khía cạnh [{col:8s}]: Accuracy = {acc:.4f} | F1-Macro = {f1:.4f}")
            
        print(f"==> {set_name} Trung bình: Mean Accuracy = {np.mean(acc_list):.4f} | Mean F1-Macro = {np.mean(f1_list):.4f}")
    print("\n")

# ==========================================
# 4. THỰC NGHIỆM MÔ HÌNH BASELINE
# ==========================================
# 1. Naive Bayes
nb_model = MultiOutputClassifier(MultinomialNB())
evaluate_model(nb_model, "Multinomial Naive Bayes")

# 2. Support Vector Machine (SVM)
svm_model = MultiOutputClassifier(LinearSVC(random_state=42, max_iter=2000))
evaluate_model(svm_model, "Support Vector Machine (LinearSVC)")