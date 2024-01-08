import json
import re
import sys

def remove_unicode_characters(string):
    pattern = re.compile(r'[\u0080-\uFFFF]')
    pattern = re.compile(r'[\u0080-\uFFFF\[\&\#\d\d\d\d;...\]]')
    string = re.sub(pattern, '', string)

    return string

def clean_data(raw_data_file,cleaned_file):
    with open(raw_data_file, 'r') as json_file:
        data = json.load(json_file)['data']

    llama_data = []
    for d in data:
        if d['language'] == 'en':
            title = remove_unicode_characters(d['title'])
            des = remove_unicode_characters(d['description'])
            if len(title) != len(des):
                sys_prompt = "[INST] <> Write an appropriate description for the given title. <> \n "
                text = sys_prompt + title + " [/INST] \n\n " + des
                llama_data.append(text)

    final_train_data = {"text": llama_data}

    with open(cleaned_file, 'w') as json_file:
        json.dump(final_train_data, json_file, indent=2)


if __name__ == '__main__':
    clean_data(sys.argv[1],sys.argv[2])