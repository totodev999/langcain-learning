from langchain_community.document_loaders import WikipediaLoader
import bs4
import requests

# 東京証券取引所プライム市場上場企業一覧
url = "https://ja.wikipedia.org/wiki/%E6%9D%B1%E4%BA%AC%E8%A8%BC%E5%88%B8%E5%8F%96%E5%BC%95%E6%89%80%E3%83%97%E3%83%A9%E3%82%A4%E3%83%A0%E5%B8%82%E5%A0%B4%E4%B8%8A%E5%A0%B4%E4%BC%81%E6%A5%AD%E4%B8%80%E8%A6%A7"
res = requests.get(url)
soup = bs4.BeautifulSoup(res.text, "html.parser")

# 企業の一覧テーブルを取得
a_tags = soup.find("table",{"class":"wikitable"}).find_all("a")

# リンクのみを取得。なお、まだページがないものを除外する
target_links = []
for a_tag in a_tags:
    href:str = a_tag.get("href")
    class_name:str|None = a_tag.get("class")
    if href.startswith("/wiki/") and class_name is None:
        target_links.append(href)

# リンクをテキストに保存
with open("./wikipedia_japan_prime_market.txt", "a") as f:
    for link in target_links:
        f.write(f"https://ja.wikipedia.org{link}\n")
