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
import sys

warnings.filterwarnings("ignore")
target = str(sys.argv[2])
file_name = sys.argv[1]
f_path = f"../data/tez/{str(file_name)}/PreSynthetic.csv"
# target = "income"

df = pd.read_csv(f_path, sep=";")

val1 = df[target].unique()[0]
val2 = df[target].unique()[1]

val1_shape = df[df[target] == val1].shape[0]
val2_shape = df[df[target] == val2].shape[0]

X = df.drop(target, axis=1).values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

cgan = cGAN(out_shape=X.shape[1])

y_train = y_train.reshape(-1, 1)
pos_index = np.where(y_train == val1)[0]
neg_index = np.where(y_train == val2)[0]
# print("pos index,", pos_index)
# print("len pos index,", len(pos_index))
# print(type(X_train))
# print(X_train.iloc[pos_index])
cgan.train(X_train, y_train, pos_index, neg_index, epochs=500)

# we want to generate 19758 instances with class value 0 since that represents how many 0s are in the label of the real training set
noise = np.random.normal(0, 1, (val1_shape, 32))
sampled_labels = np.zeros(val1_shape).reshape(-1, 1)

gen_samples = cgan.generator.predict([noise, sampled_labels])

gen_df = pd.DataFrame(data=gen_samples,
                      columns=df.drop(target, 1).columns)

print("gen_df \n", gen_df)
# we want to generate 6290 instances with class value 1 since that represents how many 1s are in the label of the real training set
noise_2 = np.random.normal(0, 1, (val2_shape, 32))
sampled_labels_2 = np.ones(val2_shape).reshape(-1, 1)

gen_samples_2 = cgan.generator.predict([noise_2, sampled_labels_2])

gen_df_2 = pd.DataFrame(data=gen_samples_2,
                        columns=df.drop(target, 1).columns)

gen_df_2[target] = 1
gen_df[target] = 0
print("gen_df_2 \n", gen_df_2)

df_gan = pd.concat([gen_df_2, gen_df], ignore_index=True, sort=False)
df_gan = df_gan.sample(frac=1).reset_index(drop=True)
print("df_gan \n", df_gan)

df_gan.to_csv(f"../data/tez/{str(file_name)}/VanilliaCGAN_EncodedSyntheticData.csv", sep=";", index=False)
print("shape \n", df_gan.shape)
