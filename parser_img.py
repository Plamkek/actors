import os
from threading import Thread
from urllib.parse import urlparse
from duckduckgo_search import DDGS
import requests

RESULTS = 500
QUERIES = (
    'Tom+Hardy',
    'Keira+Knightley',
    'Margot+Robbie',
    'Ryan+Reynolds',
    'Javier+Bardem'
)

def get_results(query: str):
    i = 0
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0'
    }
    with DDGS() as ddgs:
        downloaded_images = set()
        for image in ddgs.images(query, max_results=RESULTS):
            if image['image'].lower().endswith('.jpg'):
                _, filename = os.path.split(urlparse(image['image']).path)
                if '?' in filename:
                    filename, _ = filename.split('?', maxsplit=1)

                if filename not in downloaded_images:
                    try:
                        response = requests.get(image['image'], headers=headers)
                        if response.status_code == 200:
                            i += 1
                            with open(f'DataSet/{query}/{query}_{i}.jpg', mode='wb') as file:
                                file.write(response.content)
                            downloaded_images.add(f'{filename}')
                    except Exception as e:
                        print(f"Error downloading {filename}: {e}")

threads = []
for query in QUERIES[:10]:
    try:
        os.makedirs(f'DataSet/{query}')
    except FileExistsError:
        pass
    t = Thread(target=get_results, args=(query,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
