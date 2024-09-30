import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib
# %%
import pandas as pd
def read_data():
    df = pd.read_csv(f"out.csv")
    return df

def read_sample_data():
    df = pd.read_csv(f"out.csv",nrows=50)
    return df
# df=read_data()
# %%
def explor_data(df):
    df.columns
    df.dtypes
    df.shape
    df.info()
    print(df.isna().sum())

# explor_data()
# %%

def droping_meaningless_features(df):
    try:
        if 'whois_data' in df.columns:
            df = df.drop('whois_data', axis=1)
    except:
        pass
    try:
        if 'source' in df.columns:
            df = df.drop('source', axis=1)
    except: 
        pass
    try:
        if 'url' in df.columns:
            df = df.drop('url', axis=1)  # حذف العمود النصي 'url'
    except:
        pass
    return df 

# df = droping_meaningless_features(df)
# %%
def fill_in_data(df):
    if type(df)==pd.DataFrame:
        df = df.fillna(value=-1)
        cols = df.columns.to_list()
        for col in df[cols]:
            print(f"df[{col}] the nunique value is = {df[col].nunique()}")
        return df
    else :
        raise ValueError("Parameter type must be DataFrame") 
    return

# df=fill_in_data(df)
# %%
def encoding_data(df):
    if type(df)==pd.DataFrame:
        try:
            label_encoder = LabelEncoder()
            df['starts_with_ip'] = label_encoder.fit_transform(df['starts_with_ip'])
            df['domain_has_digits'] = label_encoder.fit_transform(df['domain_has_digits'])
            df['has_internal_links'] = label_encoder.fit_transform(df['has_internal_links'])
            df['has_punycode'] = label_encoder.fit_transform(df['has_punycode'])
            return df
        except: 
            raise (f"error")

# df=encoding_data(df)
# %%
def information(df):
    for column in df.select_dtypes(include=['object']):
        df[column], _ = pd.factorize(df[column])
    inf=df.info()
    return inf
# df ,inf =information(df)
# print(inf)
# %%
def split_data(df):
    X = df.drop(['label'], axis=1)
    y =  df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = BaggingClassifier()
    model.fit(X_train, y_train)
    return model
# X_train, X_test, y_train, y_test=split_data(df)
# model=train_model(X_train, y_train)
# %%
def model_score(model,X_train, X_test, y_train, y_test):

    model.score(X_train, y_train)
    model.score(X_test, y_test)
    predict = model.predict(X_test)
    r2 = r2_score(y_test,predict)
    # print(r2)
    # print("______________")
    # print(X_test.head(2))
    return r2 ,predict
   
# %%
import re
import math
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# حساب التعقيد الإحصائي (Entropy)
def calculate_entropy(s):
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return - sum([p * math.log2(p) for p in prob])

def extract_url_features(url, source=0, domain_age_days=-1):
    # تحليل الرابط
    parsed_url = urlparse(url)
    
    # استخراج الميزات
    url_length = len(url)
    starts_with_ip = bool(re.match(r'^https?://\d+\.\d+\.\d+\.\d+', url))
    url_entropy = calculate_entropy(url)
    has_punycode = 'xn--' in url
    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    digit_letter_ratio = digits / letters if letters > 0 else 0
    dot_count = url.count('.')
    at_count = url.count('@')
    dash_count = url.count('-')
    tld_count = len(parsed_url.netloc.split('.')) - 1
    domain_has_digits = any(char.isdigit() for char in parsed_url.netloc)
    subdomain_count = len(parsed_url.netloc.split('.')) - 2 if len(parsed_url.netloc.split('.')) > 2 else 0
    nan_char_entropy = calculate_entropy(parsed_url.netloc)
    has_internal_links = 0  # تحتاج إلى HTML لتحليل الروابط الداخلية

    features = {
        'url': [url],    
        'url_length': [url_length],
        'starts_with_ip': [int(starts_with_ip)],
        'url_entropy': [url_entropy],
        'has_punycode': [int(has_punycode)],
        'digit_letter_ratio': [digit_letter_ratio],
        'dot_count': [dot_count],
        'at_count': [at_count],
        'dash_count': [dash_count],
        'tld_count': [tld_count],
        'domain_has_digits': [int(domain_has_digits)],
        'subdomain_count': [subdomain_count],
        'nan_char_entropy': [nan_char_entropy],
        'has_internal_links': [has_internal_links],
        'domain_age_days': [domain_age_days]
    }
    
    new_data = pd.DataFrame(features)
    return new_data
# ___________________________________________________________________________________________________--
def extract_url_features2(urls, source=0, domain_age_days=-1):
    # تقسيم السلسلة إلى روابط بناءً على المسافة الفاصلة
    url_list = urls.split()

    all_features = []

    for url in url_list:
        # تحليل الرابط
        parsed_url = urlparse(url)

        # استخراج الميزات
        url_length = len(url)
        starts_with_ip = bool(re.match(r'^https?://\d+\.\d+\.\d+\.\d+', url))
        url_entropy = calculate_entropy(url)
        has_punycode = 'xn--' in url
        letters = sum(c.isalpha() for c in url)
        digits = sum(c.isdigit() for c in url)
        digit_letter_ratio = digits / letters if letters > 0 else 0
        dot_count = url.count('.')
        at_count = url.count('@')
        dash_count = url.count('-')
        tld_count = len(parsed_url.netloc.split('.')) - 1
        domain_has_digits = any(char.isdigit() for char in parsed_url.netloc)
        subdomain_count = len(parsed_url.netloc.split('.')) - 2 if len(parsed_url.netloc.split('.')) > 2 else 0
        nan_char_entropy = calculate_entropy(parsed_url.netloc)
        has_internal_links = 0  # تحتاج إلى HTML لتحليل الروابط الداخلية

        features = {
            'url': url,    
            'url_length': url_length,
            'starts_with_ip': int(starts_with_ip),
            'url_entropy': url_entropy,
            'has_punycode': int(has_punycode),
            'digit_letter_ratio': digit_letter_ratio,
            'dot_count': dot_count,
            'at_count': at_count,
            'dash_count': dash_count,
            'tld_count': tld_count,
            'domain_has_digits': int(domain_has_digits),
            'subdomain_count': subdomain_count,
            'nan_char_entropy': nan_char_entropy,
            'has_internal_links': has_internal_links,
            'domain_age_days': domain_age_days
        }

        # إضافة ميزات الرابط إلى القائمة
        all_features.append(features)

    # إنشاء إطار بيانات يحتوي على جميع الميزات للروابط المختلفة
    df = pd.DataFrame(all_features)
    return df
# ___________________________________________________________________________________________________--
# Function to predict new data
def predict_new_data(url=""):
    try:
        url_features = extract_url_features(url)
        encoded_data = encoding_data(url_features)
        cleaned_data = droping_meaningless_features(encoded_data)
        
        model = load_model() 
        prediction = model.predict(cleaned_data)
        return prediction, cleaned_data
    except Exception as e:
        return f"Error in prediction: {e}", f"URL: {url}"

    
def results(url="http://example.com/test-url"):
    pn,new_data=predict_new_data(url)
    print("Prediction for the new URL:", pn)
    if pn == 0:
        print("URL is predicted as safe.")
    else:
        print("URL is predicted as malicious.")
    return pn,new_data
    
# pn, new_data = results()
# %%
def calculate_main_error(y_test , predict):
    Main_Absolute_Error = mean_absolute_error(y_test,predict)
    return Main_Absolute_Error
# calculate_main_error()
# %%
def conf_matrix(y_test , predict):
    cm = confusion_matrix(y_test,predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
# conf_matrix()
# %%
import joblib
# Function to save the trained model
def save_model(model, filename="uRL_trained_model.pkl"):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")
# %%
# Function to load the model
import os 
def load_model(filename="uRL_trained_model.pkl"):
    model = joblib.load(filename=filename)
    full_path = os.path.abspath(filename)
    print(f"Model loaded from {full_path}")
    return model
 
from sklearn.metrics import accuracy_score
# Function to evaluate the model's accuracy
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    return accuracy

def model_prepare():
    # if not df:
    df=read_data()  
    df= droping_meaningless_features(df)
    df=fill_in_data(df) 
    df=encoding_data(df) 
    inf =information(df) 
    X_train , X_test , y_train, y_test = split_data(df) 
    model=train_model(X_train, y_train)
    return model


def retrain_model():
    model = model_prepare()
    return model

