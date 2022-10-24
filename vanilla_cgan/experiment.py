import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, \
    accuracy_score, balanced_accuracy_score, classification_report, \
    plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from cgan_network import cGAN
import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('adult.csv')
df.head()

le = preprocessing.LabelEncoder()
for i in ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country',
          'income']:
    df[i] = le.fit_transform(df[i].astype(str))

print("original df ", df.head())

df.income.value_counts()

scaler = StandardScaler()

X = scaler.fit_transform(df.drop('income', 1))
y = df['income'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lgb_1 = lgb.LGBMClassifier()
lgb_1.fit(X_train, y_train)

y_pred = lgb_1.predict(X_test)

# evaluation
print(classification_report(y_test, y_pred))
plot_confusion_matrix(lgb_1, X_test, y_test)
plt.show()

cgan = cGAN()

y_train = y_train.reshape(-1, 1)
pos_index = np.where(y_train == 1)[0]
neg_index = np.where(y_train == 0)[0]
cgan.train(X_train, y_train, pos_index, neg_index, epochs=500)

# we want to generate 19758 instances with class value 0 since that represents how many 0s are in the label of the real training set
noise = np.random.normal(0, 1, (19758, 32))
sampled_labels = np.zeros(19758).reshape(-1, 1)

gen_samples = cgan.generator.predict([noise, sampled_labels])

gen_df = pd.DataFrame(data=gen_samples,
                      columns=df.drop('income', 1).columns)

print("gen_df \n", gen_df)
# we want to generate 6290 instances with class value 1 since that represents how many 1s are in the label of the real training set
noise_2 = np.random.normal(0, 1, (6290, 32))
sampled_labels_2 = np.ones(6290).reshape(-1, 1)

gen_samples_2 = cgan.generator.predict([noise_2, sampled_labels_2])

gen_df_2 = pd.DataFrame(data=gen_samples_2,
                        columns=df.drop('income', 1).columns)

gen_df_2['income'] = 1
gen_df['income'] = 0
print("gen_df_2 \n", gen_df_2)

df_gan = pd.concat([gen_df_2, gen_df], ignore_index=True, sort=False)
df_gan = df_gan.sample(frac=1).reset_index(drop=True)
print("df_gan \n", df_gan)

X_train_2 = df_gan.drop("income", 1)
y_train_2 = df_gan['income'].values

lgb_1 = lgb.LGBMClassifier()
lgb_1.fit(X_train_2, y_train_2)

y_pred = lgb_1.predict(X_test)

print(classification_report(y_test, y_pred))
plot_confusion_matrix(lgb_1, X_test, y_test)
plt.show()
