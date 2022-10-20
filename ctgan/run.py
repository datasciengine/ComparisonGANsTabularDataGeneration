from ctgan import CTGAN
import warnings
import pandas as pd
import sys

warnings.filterwarnings("ignore")
target = str(sys.argv[2])
file_name = sys.argv[1]
f_path = f"../data/tez/{str(file_name)}/PreSynthetic.csv"
# target = "income"
import datetime

print(str(datetime.datetime.now()))
real_data = pd.read_csv(f_path, sep=";")
# discrete_columns = ["children", "sex", "smoker", "region"]
# Names of the columns that are discrete
# discrete_columns = [
#     "education-num",
#     "workclass",
#     "education",
#     "marital-status",
#     "occupation",
#     "relationship",
#     "race",
#     "sex",
#     "native-country",
#     "label"
# ]

discrete_columns = [
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio",
    "gender"
]
ctgan = CTGAN(epochs=500, batch_size=5000)

ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(real_data.shape[0])
# synthetic_data.to_csv("fake.csv", index=False, sep=";")
synthetic_data.to_csv(f"../data/tez/{str(file_name)}/CTGAN_EncodedSyntheticData.csv",
                      sep=";",
                      index=False)

print(str(datetime.datetime.now()))
