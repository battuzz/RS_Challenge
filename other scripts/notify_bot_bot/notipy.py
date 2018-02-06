import json
import requests

TOKEN = '455483140:AAES75gXEnEowExVUITHWNOq4rEsgno03gI'
root_url = 'https://api.telegram.org/bot{}/'.format(TOKEN)

CHAT_NUM = '-194097364'

def notify(message):
    url = root_url + "sendMessage?text={}&chat_id={}".format(message, CHAT_NUM)
    try:
        requests.get(url)
    except:
        print('Error sending the notification.')
