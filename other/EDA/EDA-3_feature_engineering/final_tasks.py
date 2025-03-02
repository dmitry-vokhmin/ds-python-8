import pandas as pd
import category_encoders as ce
from matplotlib import pyplot as plt
from sklearn import preprocessing
import seaborn as sns

heart = pd.read_csv('heart.csv')
heart['old'] = heart['age'].apply(lambda x: 1 if x > 60 else 0)

pressure_mapping = {
    'male': {
        (0, 20): 123,
        (21, 30): 126,
        (31, 40): 129,
        (41, 50): 135,
        (51, 60): 142,
        (61, float('inf')): 142
    },
    'female': {
        (0, 20): 116,
        (21, 30): 120,
        (31, 40): 127,
        (41, 50): 137,
        (51, 60): 144,
        (61, float('inf')): 159
    }
}

def get_trestbps_mean(age, sex):
    # Определяем маппинг по полу
    gender = 'male' if sex == 1 else 'female'

    # Ищем подходящий диапазон возраста
    for age_range, pressure in pressure_mapping[gender].items():
        if age_range[0] <= age <= age_range[1]:
            return pressure

heart['trestbps_mean'] = heart.apply(lambda row: get_trestbps_mean(row['age'], row['sex']), axis=1)

ord_encoder = ce.OneHotEncoder(cols=['cp', 'ca', 'restecg', 'slope', 'thal'])
extra = ord_encoder.fit_transform(heart[['cp', 'ca', 'restecg', 'slope', 'thal']])
heart = pd.concat([heart, extra], axis=1)
heart = heart.drop(columns=['cp', 'ca', 'restecg', 'slope', 'thal'])

# инициализируем нормализатор RobustScaler
r_scaler = preprocessing.RobustScaler()
col_names = list(heart.columns)

# копируем исходный датасет
heart_r = r_scaler.fit_transform(heart)

heart_r = pd.DataFrame(heart_r, columns=col_names)

# смотрим описательные статистики, ответ 0.816232
heart_r.describe()

plt.figure(figsize=(20, 15))
sns.heatmap(heart.corr(numeric_only=True), annot=True)
plt.show()
print(1)