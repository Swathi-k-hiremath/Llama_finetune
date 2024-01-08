import http.client, urllib.parse
import json
import sys

conn = http.client.HTTPConnection('api.mediastack.com')

params = urllib.parse.urlencode({
    'access_key': 'df1c1027bc3dd38b6cddb5e53a1ec1da',
    'categories': '-general,-sports',
    'sort': 'published_desc',
    'limit': 100
    })

conn.request('GET', '/v1/news?{}'.format(params))

if __name__ == '__main__':
    res = conn.getresponse()
    data = res.read()
    text_data= data.decode('utf-8')
    dictionary = json.loads(text_data)

    file_path = sys.argv[1]

    # Write the JSON data to the file
    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=2)