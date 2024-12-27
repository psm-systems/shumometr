import telebot
import soundfile as sf
import os
import warnings
from ast_master import model
warnings.filterwarnings("ignore") #Обход всплывающего сообщения Cuda

#                   Бот запускается с помощью watchdog supervisor
#                   Команда для запуска:
#                   supervisord -c supervisor.conf &
#                   Конфигурация watchdog в файле supervisor.conf

# Установка локализации
# locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')

# Инициализация бота
# Необходимо добавить token в виде строки
bot = telebot.TeleBot(token='', threaded=False)


# реакция на команду /start
@bot.message_handler(commands=['start'])
def start_bot(msg):
    bot.send_message(msg.chat.id, f"Привет! Я бот, который определяет источники звуков "
                                  f"в голосовых сообщениях. Могу расслышать от писка комара до шума стройки."
                                  f" Используй меня, если требуется узнать шумовую обстановку!")


# реакция на команду /tst для проверки работоспособности
@bot.message_handler(commands=['tst'])
def tst_bot(msg):
    bot.send_message(msg.chat.id, f"🤖")


@bot.message_handler(content_types=['voice'])
def voice_processing(message):
    if not os.path.isdir('../image_message'):
        os.mkdir('../image_message')
    if not os.path.isdir('../voice_message'):
        os.mkdir('../voice_message')
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # запись временного голосового сообщения в файл с именем id чата в формате ogg
    with open(f'../voice_message/{message.chat.id}.ogg', 'wb') as new_file:
        new_file.write(downloaded_file)

    # конвертация голосового сообщения в формат wav
    data, samplerate = sf.read(f'../voice_message/{message.chat.id}.ogg')
    sf.write(f'../voice_message/{message.chat.id}.wav', data, samplerate)

    # удаление временного голосового сообщения
    os.remove(f'../voice_message/{message.chat.id}.ogg')

    # что-то что будет дергать модель
    res_message = model.ast_model_main(path=f'../voice_message/', name=f'{message.chat.id}')

    # отправляем сообщение пользователю
    bot.send_message(message.chat.id, res_message)
    with open(f'../image_message/{message.chat.id}.png', 'rb') as image:
        bot.send_photo(message.chat.id, image)

    # удаляем временные файлы
    os.remove(f'../voice_message/{message.chat.id}.wav')
    os.remove(f'../image_message/{message.chat.id}.png')


# Запуск бота
bot.infinity_polling(timeout=6000)
