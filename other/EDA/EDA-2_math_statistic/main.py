import pandas as pd
import statistics
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import matthews_corrcoef
import seaborn as sns

df = pd.read_csv('wine_cleared.csv', index_col=0)
print('наименьший балл за вино', df['points'].min())
print('наименьшую цена за бутылку', df['price'].min())

plt.subplot(1, 2, 1)  # задаем сетку рисунка количество строк и столбцов
stats.probplot(df['points'], plot=plt)  # qq plot

plt.subplot(1, 2, 2)  # располагаем второй рисунок рядом
plt.hist(df['points'])  # гистограмма распределения признака

plt.tight_layout()  # чтобы графики не наезжали другу на друга, используем tight_layout

print('коэффициент корреляции Спирмена для признаков point и price', df['points'].corr(df['price'], method='spearman'))
print('коэффициент корреляции Кендалла для признаков point и price', df['points'].corr(df['price'], method='kendall'))

x = [+1, -1, +1, +1] # список значений признака х
y = [+1, +1, +1, -1] # список значений признака y

res = matthews_corrcoef(x, y) # рассчитаем коэффициент корреляции Мэтьюса
print('коэффициента корреляции Мэтьюса', res)

df = pd.read_csv('model.csv')
correlation_matrix = df.corr(numeric_only=True)

print('матрица корреляций', correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()
sns.scatterplot(data=df, x="Waist/Hip", y="Waist")
plt.show()
sns.pairplot(df)
plt.show()
