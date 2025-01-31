## shumometr
# Телеграмм бот. Принимает голосовые сообщения и определяет тип шумов в записи.

Крупные города не прекращают шуметь ни на мгновение. Звуки окружают человека повсюду: на улицах, в общественном транспорте, на работе, в магазинах, ресторанах и даже дома.
Шумовое загрязнение — это раздражающий шум антропогенного происхождения, который нарушает привычную жизнедеятельность человека и животных. К антропогенным факторам относят звуки движущегося транспорта, строительной техники, работающих производств и различных механизмов. Для каждого мегаполиса или крупного города шумовое загрязнение - серьезная проблема, негативно влияющая на окружающую среду и качество жизни.

На основе пред обученной модели группы студентов из Кембриджа был создан telegram-bot "Шумометр", задачей которого является оценить тот или иной район по составу шума.
Бот принимает голосовые сообщения, записанные в приложение telegram, обрабатывает эти данные и предоставляет пользователю визуальный и текстовый отчет о компонентах, составляющих голосовое сообщение с шумом.

Бот может быть полезен для предварительной оценки качества вида шума. Например, это может быть актуально при покупке/съёме жилья.
Целевая аудитория – это будущие покупатели квартир, агентства недвижимости, сами застройщики, а также различные экологические организации.

В своей основе проект использует свободно распространяемые библиотеки:
-telebot - для создания серверной части кода для бота
-pytorch - для обучения модели
-pandas - для обработки полученных данных от модели
-plotly.express - для создания инфографики
-numpy - для расчетов и работы с массивами

За основную часть модели была взята модель из репозитория https://github.com/YuanGongND/ast/tree/master

Модель работает на спектральном анализе аудиофайлов wav формата. Она содержит официальную реализацию PyTorch Audio Spectrogram Transformer (AST).
AST — это первая модель классификации звука без свертки, основанная исключительно на внимании, которая поддерживает ввод переменной длины и может применяться для различных задач.
Так как telegramm по умолчанию используев голосовые сообщения в расширение .ogg, программная часть бота осуществляет конвретацию файлов с расширением .ogg в расширение .wav для совместимости работы с моделью, работающей с файлами в формате .flac, .wav, .mp3

# Установка
Установка осуществляется копированием репозитория

`git clone https://github.com/psm-systems/shumometr.git`

и установкой всех зависимостей в соостветствие с их версиями (находятся в файле requirements.txt)

`pip install -r requirements.txt`

Далее для корректной работы вашего телеграмм бота требуется заменить token в модуле telegram.py в качестве строки

`bot = telebot.TeleBot(token='111111111:XXXXXXXXXXXXXXXXXXXXX', threaded=False)`

Следующим шагом будет установка watchdog - Supervisor: A Process Control System. Supervisor полностью реализован на python3, поэтому не требует дополнительных программ и может быть установлен с помощь команды

`pip install supervisor`


Теперь осталось установить конвертер аудио файлов. Это делается с помощью модуля Soundfile и выполняется командой

`pip install soundfile`

Бот готов к работе и запуску. Для запуска требуется перейти в директорию ast_master в вашем окружение.

> При первом запуске и обработки первого голосового сообщения требует значительно больше времени, так как бот суачивает пред обученную модель


1) Для запуска с помощью Supervisor необходимо ввести команду

`supervisord -c supervisor.conf &`

2) Для запуска без Supervisor необходимо ввести команду

`python3 ./telegram.py`

# Использование
Бот фильтрует все команды кроме:
/start - запуск бота и вывод информации о боте
/tst - простой тест работы бота (ping из телеграмма)

Также бот НЕ реагирует ни на какие медиа файлы и текстовые сообщения. Бот принимает только голосовые сообщения для их анализа.

Не рекомендуется записывать голосовые сообщения и параллельно вести беседу. В виду спицифики работы telegramm голос будет распознан и усилен, а посторонние шумы будут заглушены. Это приведет к некорректной работе модели, неверному анализу и ошибочному выводу.

# Дополнительно
> Возможна некорректная работа модуля Soundfile. Исправляется переустановкой модуля pip install --force-reinstall soundfile

> Бот тестировался на системе Windows 11 Version 10.0.22621.3155]
