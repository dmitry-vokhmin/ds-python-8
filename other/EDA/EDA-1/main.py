import pandas as pd
from pandas import DataFrame
from ydata_profiling import ProfileReport

data = pd.read_csv('wine.csv')
print('Сколько всего дегустаторов приняло участие в винных обзорах?', data['taster_name'].nunique())
print(data['price'].max())
print('все числовые признаки', data.select_dtypes(include=['number']).columns.tolist())
print('наличие дублирующихся винных обзоров', data[data.duplicated()])
data = data.drop_duplicates()
print('наличие пропусков в данных', data.isna().any())
missing_values = data.isnull().mean() * 100
data: DataFrame = data.loc[:, missing_values < 30]

data['designation'] = data['designation'].fillna('unknown')
data['region_1'] = data['region_1'].fillna('unknown')
data['taster_name'] = data['taster_name'].fillna('unknown')
data['taster_twitter_handle'] = data['taster_twitter_handle'].fillna('unknown')

# признаки с маленьким количеством пропусков заменим на самые частовречающиеся значения
data['country'] = data['country'].fillna('US')
data['price'] = data['price'].fillna(data['price'].mean())
data['province'] = data['province'].fillna('California')
data['variety'] = data['variety'].fillna('Pinot Noir')

profile = ProfileReport(data, title="Pandas Profiling Report")
profile.to_file("your_report.html")
