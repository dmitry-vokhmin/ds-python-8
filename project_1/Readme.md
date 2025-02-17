# Проект: Анализ резюме с сайта HeadHunter

## Цели и задачи проекта

Цель проекта — проведение анализа данных, полученных из резюме соискателей с платформы HeadHunter, с использованием методов предобработки данных и визуализации. Основные задачи:
1. Изучить и очистить данные от пропусков, дубликатов и выбросов.
2. Преобразовать исходные данные для последующего анализа, включая извлечение полезных признаков.
3. Провести исследование зависимостей между признаками и сделать выводы.
4. Подготовить данные для возможных задач построения модели прогнозирования заработной платы.

---

## Этапы работы

### 1. **Исследование структуры данных**
- Загрузка данных с использованием библиотеки Pandas.
- Изучение структуры данных (типы столбцов, наличие пропусков, анализ категорий).
- Вывод статистической информации о выбираемых признаках.

### 2. **Преобразование данных**
- Извлечение категорий из сложных текстовых столбцов:
  - Признак уровня образования и вуза.
  - Пол и возраст соискателя.
  - Опыт работы в месяцах.
  - Город проживания и готовность к переезду/командировкам.
  - Занятость и график работы, преобразованные в бинарные признаки.
- Приведение заработной платы к рублям на основании данных о курсе валют.

### 3. **Исследовательский анализ данных (EDA)**
- Построение распределений для анализа признаков, включая выявление выбросов.
- Вывод зависимости заработной платы от:
  - уровня образования;
  - города проживания;
  - возраста и опыта работы.
- Построение тепловых карт и других визуализаций.

### 4. **Очистка данных**
- Удаление полных дубликатов и обработка пропусков.
- Удаление выбросов:
  - заработная плата выше 1 млн руб. или ниже 1 тыс. руб.;
  - несоответствие между возрастом и опытом работы (опыта больше, чем возраст);
  - преклонный возраст или возраст, не подходящий для поиска работы.
  
### 5. **Выводы**
- Анализ взаимосвязей между признаками и их значимость в прогнозировании заработной платы.
- Формирование чистого датасета, готового к построению моделей машинного обучения.

---

## Использованные инструменты

- **Язык программирования**: Python.
- **Библиотеки и фреймворки**:
  - Pandas, NumPy — для работы с данными.
  - Matplotlib, Seaborn, Plotly — для построения диаграмм и графиков.
- **Дополнительные файлы**: данные о курсе валют с внешнего ресурса.

---

## Выводы

1. **Анализ возраста**:
   - Большинство соискателей находятся в диапазоне от 14 до 49 лет.
   - Были выявлены аномальные значения возраста, например, 100 лет.

2. **Опыт работы**:
   - У большинства соискателей опыт работы находится в диапазоне от 1 до 299 месяцев, мода — 81 месяц.
   - Некоторые данные опыта работы считались аномальными и удалены.

3. **Распределение зарплат**:
   - Большинство соискателей ожидают зарплату в диапазоне от 12 000 до 180 000 руб.
   - В городах Москва и Санкт-Петербург наблюдаются самые высокие медианные зарплаты.

4. **Образование**:
   - Наибольшие зарплатные ожидания у соискателей с высшим образованием.
   - Признак уровня образования имеет значительное влияние на предсказание заработной платы.

5. **Признаки готовности к переездам и командировкам**:
   - Соискатели, готовые к переезду и командировкам, ожидают более высоких зарплат.

6. **Тепловая карта — зарплата по возрасту и образованию**:
   - С возрастом рост зарплат остановлен для номинально низких уровней образования, а для высшего продолжается.
