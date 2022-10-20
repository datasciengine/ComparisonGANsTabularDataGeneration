from ctgan import CTGAN
from ctgan import load_demo
import warnings

warnings.filterwarnings("ignore")
real_data = load_demo()

# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

ctgan = CTGAN(epochs=1)

print("yes")
ctgan.fit(real_data, discrete_columns)
print("okk")
# Create synthetic data
synthetic_data = ctgan.sample(1000)
synthetic_data.to_csv("fake.csv", index=False, sep=";")

print(synthetic_data.head(10))
print("___")
print(synthetic_data.info())
print("done!")
