import requests
from bs4 import BeautifulSoup

url = "https://steamcommunity.com/market/listings/730/AK-47%20%7C%20Leet%20Museo%20%28Field-Tested%29"

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 获取第一个只有“normal_price”类的<span>元素
result = soup.find('span', {'class': 'market_listing_price market_listing_price_with_fee'})

# 提取<span>元素的文本内容
if result:
    price = result.text.strip()
    print(price)
else:
    print("未找到指定元素")