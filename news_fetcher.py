import os
from newsdataapi import NewsDataApiClient

NEWS_API_KEY = "pub_869e776e5a3748a5979d0b3790562cb1"

# Init client
api = NewsDataApiClient(apikey=NEWS_API_KEY)

def fetch_live_news(query="AI", max_articles=50):
    articles = []
    next_page = None

    while len(articles) < max_articles:
        response = api.news_api(
            q=query,
            language="en",
            page=next_page
        )
        if "results" not in response:
            break
        articles.extend(response["results"])
        next_page = response.get("nextPage")
        if not next_page:
            break

    return articles[:max_articles]
