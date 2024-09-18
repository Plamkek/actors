import telebot
import timm
from PIL import Image
from telebot import types
import requests
import torch
import torchvision.transforms as transforms


f = open("token.txt")
API_TOKEN = f.read().strip()
f.close()
bot = telebot.TeleBot(API_TOKEN)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('resnet50.a1_in1k', pretrained=False).to(device)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5).to(device)
model.load_state_dict(torch.load('model.pt', map_location=device))
model.eval()

with open('classes.txt', encoding='utf-8', mode='r') as file:
    classes = [s.strip() for s in file.readlines()]


# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, """\
Напиши /cat чтобы получить фотку кота или отправь фотку актера
""")


def get_cat() -> str:
    contents = requests.get('https://cataas.com/cat?json=true').json()
    url = contents['_id']
    return 'https://cataas.com/cat?id=' + url


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    file_path = file_info.file_path
    downloaded_file = bot.download_file(file_path)

    with open('image.jpg', 'wb') as qwe:
        qwe.write(downloaded_file)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open('image.jpg')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0] * 100
        prob, predicted_indices = torch.topk(probabilities, k=5)

    text = ''
    for prob, ind in zip(prob, predicted_indices):
        cl = classes[ind]
        pr = prob.item()
        text += f"Это {cl} с вероятностью {pr:.2f}%\n"

    bot.reply_to(message, text)


@bot.message_handler(commands=['cat'])
def cat_message(message: types.Message):
    cid = message.chat.id
    bot.send_chat_action(cid, 'upload_photo')
    q = get_cat()
    bot.send_photo(cid, q)


bot.infinity_polling()

