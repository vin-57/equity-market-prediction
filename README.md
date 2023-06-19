# Веб-приложение для прогнозирования котировок акций с использованием машинного обучения
Веб-приложение для прогнозирования котировок акций на основе машинного обучения и анализа настроений в сообщениях в СМИ (ключи API включены в код).
Внешний интерфейс веб-приложения основан на Streamlit и Wordpress. 
Приложение прогнозирует цены на акции на следующие семь дней для акций компаний, котирующихся на NASDAQ или NSE, вводимых пользователем. 
Прогнозы делаются с использованием трех алгоритмов: ARIMA, LSTM, Linear Regression. 
Веб-приложение объединяет прогнозируемые цены на следующие семь дней с анализом настроений сообщений в СМИ, чтобы дать рекомендацию о том, будет ли цена расти или падать.

# Примечание
Файл Wordpress был перемещен из репозитория из-за превышения квоты Github LFS. 

# Скриншоты

![Иллюстрация к проекту](https://github.com/jon/coolproject/raw/master/image/image.png)

![Image alt](https://github.com/{username}/{repository}/raw/{branch}/{path}/image.png)

# Структура файлов и каталогов
screenshots - Скриншоты веб-приложения
wordpress.sql - база данных wordpress
app.py — файл конфигурации для приложения Streamlit.
sa.py — основной модуль машинного обучения
data.csv и data2.csv - набор данных и структура сообщений из СМИ о компаниях, котирующихся на бирже, для целей анализа сентимента рынка по ним

# Используемые технологии
Wordpress
Streamlit
Tensorflow
Keras
Yahoo Finance
Alphavantage
Scikit-Learn
Python
PHP

#  Ресурсы
ЯнДЕКС ОБЛАКО
- Платформа Intel Ice Lake
- Гарантированная доля vCPU 100%
- vCPU 2
- RAM 4 ГБ
- Объём дискового пространства 50 ГБ

STREAMLIT CLOOD
- ограничения по RAM 1ГБ



# Установка и использование
- создана виртуальнуя машина на платформе Яндекс.Облако
- на непрерываемой виртуальной машине установлены операционную систему семейства Linux, веб-сервер Apache, СУБД MySQL и интерпретатор PHP
- установлен статический адрес машины [158.160.110.213](http://158.160.110.213/)
- настроен phpmuadmin и wordpress, подготовлены базы данных 
- настроен хостинг, DNS и SSL на домене www.finpredict.ru [www.finpredict.ru](https://www.finpredict.ru)
- настроен streamlit
- осуществлен деплой основного модуля прогнозирования на streamlit


# Этапы работ над проектом
1) работа со стилем кода:
- включен в потом работ GitHub Actions и настроен линтер flake8
- код форматирован в соответствии с рекомендованным стилем

2) применены методы продвинутого уровня командной разработки
- участники команды добавлены в качестве коллабораторов в репозиторий
- каждый участник создал новую ветку в репозитории и внес в нее изменения
- каждый участник создал pull request для объединения кода из новой ветки в ветку main

3) осуществлен код-ревью 
- для каждого pull request два коллаборатора провели код-ревью
- проведен рефакторинг, высказаны идеи по улучшению качества кода, которые имеет смысл реализовать (в частности, использование Streamlit, временный отказ от использования Flask и Gunicorn, оформлены коммиты
- на основе результатов проведения код ревью хозяин репозитория выполнил объдинение pull request

Таким образом, форматирован код в соответствии с рекомендованным стилем, устранена повторяемость кода, 
обработаны ошибки при работе программы, разработаны тесты, программа работоспособна.


* * *

# Авторы
Иванов Иван
[Github](https://github.com/vin-57)
Github:https://github.com/vin-57
Смирнов Антон
[Github](https://github.com/smirnovanton90)
