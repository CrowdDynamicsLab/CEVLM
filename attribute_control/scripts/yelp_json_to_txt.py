"""Convert YELP data from json to txt format"""
import json

with open('yelpdata.json', 'r') as f:
    yelp_data = f.readlines()

yelp_text = [json.loads(l)['text'] for l in yelp_data]
with open('yelpdata.txt', 'w+') as f:
    for text in yelp_text:
        f.write(text + '\n')

