import telebot
import config
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained('/Users/zinuret/Desktop/text-complexity-bot/best model')  # Укажите путь к папке с токенизатором

# Загрузка модели и весов
model = AutoModelForSequenceClassification.from_pretrained('/Users/zinuret/Desktop/text-complexity-bot/best model')  # Укажите путь к файлу с моделью


bot = telebot.TeleBot(config.TOKEN)

@bot.message_handler(commands=['start'])
def welcome(message):
    bot.send_message(message.chat.id, "Добро пожаловать, {0.first_name}!\nВведите текст для оценки сложности".format(message.from_user), parse_mode='html')

@bot.message_handler(content_types=['text'])
def evaluate(message):
    user_text = message.text

    # Tokenize the user's text
    tokens = tokenizer(user_text, padding=True, truncation=True, return_tensors='pt')
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    # Use the model to make predictions
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    # Получение списка значений из тензора
    predictions = output.logits.squeeze().tolist()

    # Извлечение оценки сложности (первого элемента списка, если их больше)
    complexity_score = predictions[0]

    # Отправка оценки обратно пользователю
    bot.send_message(message.chat.id, f"Сложность текста: {complexity_score:.2f}")


bot.polling(none_stop=True)