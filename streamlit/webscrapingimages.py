import requests
from bs4 import BeautifulSoup
import os

def get_images_links(searchTerm, num_images=60):
    try:
        search_url = f"https://www.google.com/search?q={searchTerm}&tbm=isch"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        img_tags = soup.find_all('img')

        img_urls = []
        for img in img_tags:
            if img.get('src') and img['src'].startswith("http"):
                img_urls.append(img['src'])
            if len(img_urls) >= num_images:
                break

        if len(img_urls) < num_images:
            print(f"Only found {len(img_urls)} images for search term '{searchTerm}'.")

        return img_urls

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def download_images(img_urls, download_folder='images'):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    for i, img_url in enumerate(img_urls):
        try:
            img_data = requests.get(img_url).content
            with open(os.path.join(download_folder, f"image_{i + 1}.jpg"), 'wb') as img_file:
                img_file.write(img_data)
            print(f"Downloaded image {i + 1}/{len(img_urls)}")
        except Exception as e:
            print(f"Could not download image {img_url}: {e}")

classname= [
        'Aloo Gobi', 'Aloo Matar', 'Aloo Methi', 'Aloo Tikki', 'Apple', 'Bhindi Masala',
        'Biryani', 'Boiled Egg', 'Bread', 'Burger', 'Butter Chicken', 'Chai', 'Chicken Curry',
        'Chicken Tikka', 'Chicken Wings', 'Chole', 'Daal', 'French Fries', 'French Toast', 'Fried Egg',
        'Kadhi Pakora', 'Kheer', 'Lobia Curry', 'Omelette', 'Onion Pakora', 'Onion Rings', 'Palak Paneer',
        'Pancakes', 'Paratha', 'Rice', 'Roti', 'Samosa', 'Sandwich', 'Spring Rolls', 'Waffles', 'White Rice'
    ]

for i in classname:
    img_urls = get_images_links(i)
    download_images(img_urls)
