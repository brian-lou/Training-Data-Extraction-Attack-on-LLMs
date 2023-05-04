import requests
import os

API_KEY = os.environ.get("API_KEY")
SEARCH_ENGINE_ID = os.environ.get("SEARCH_ENGINE_ID")
SEARCH_TERMS = ["Through teaching and research, we educate people who will contribute to society and develop knowledge that will make a difference in the world.", "The last time, I felt like this, in a cinema, I was six years old and I was watching Star Wars. I never imagined, I would ever find that feeling again in a cinema. That sense of being transported to another world. The opening sequence took my breath away and I never got it back. Not even at the end - which left my head spinning. It is a beautiful film with soul, wit, charm, style and love. It is simply outrageous! Bold and fantastic and fantastical. I am a straight man but my love for Ryan Gosling could change all that. He's a melancholy genius and Emma Stone is our muse. This film defies genre. It is a masterpiece. I urge you to see it. I was lucky enough to see it at the BFI London Film Festvial. It has been five days since I saw La La Land and I am still thinking about it and singing the haunting refrain that plays with your soul. I mean it gets in there - that music - the music of the firmament. Flying still, dreaming still... thank you Damien."]

for query in SEARCH_TERMS:
    url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q=%22{query}%22"
    response = requests.get(url)
    results = response.json()

    if results.get('items'):
        print(f"Results found for: '{query}'\n")
        for item in results['items']:
            print(f"Title: {item['title']}")
            print(f"Link: {item['link']}\n")
    else:
        print(f"No results found for: '{query}'\n")