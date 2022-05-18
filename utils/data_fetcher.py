! pip install instaloader

from google.colab import drive
drive.mount('/content/gdrive')

from instaloader import *
import time
import os
from pathlib import Path


bot = instaloader.Instaloader(
    download_videos = False, download_video_thumbnails=False,
    download_comments = False, save_metadata=False
)

list_of_tags = ['instagood', 'love', 'fashion', 'photooftheday', 'art', 'photography', 'beautiful', 'picoftheday', 'nature',
                'happy', 'cute', 'travel', 'style', 'instadaily', 'summer', 'beauty', 'fitness', 'food', 'instalike', 'friends',
                'family', 'life', 'music', 'lifestyle', 'design', 'motivation', 'explore', 'nofilter', 'foodporn', 'instamood',
                'artist', 'wedding', 'bestoftheday', 'workout', 'study', 'naturephotography', 'nails', 'tattoo', 'landscape',
                'blackandwhite', 'work', 'architecture', 'car', 'cat', 'dog', 'animals', 'travelphotography', 'coffee', 'adventure',
                'vscocam']

username = input("Enter your username: ")
bot.interactive_login(username=username)

def fetch_for_hashtag(hashtag, limit=1300):

    hashtags = instaloader.Hashtag.from_name(bot.context, hashtag).get_posts()
    no_of_downloads = 0
    for post in hashtags:
        try:
            if no_of_downloads == limit:
                print('Done. Sleep before next tag.')
                break
            
            str_post = str(post)
            path = Path('/content', 'gdrive', 'MyDrive', 'instproject', tag, str_post)
            path.mkdir(parents=True, exist_ok=True)

            _ = bot.download_post(post, target=path)
            if _ == True:
                no_of_downloads += 1
                print('Downloaded:', no_of_downloads)
            time.sleep(10)
        except:
            pass

for tag in list_of_tags:
    print(f'Downloading data for {tag} hashtag')
    fetch_for_hashtag(tag)
    time.sleep(120)
