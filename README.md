Download trained model from the following link 

https://drive.google.com/drive/folders/1kXO6syBxUViA1Oit3GFdCtzr8enB69gi?usp=sharing


File descriptions
- output2.json => Raw data pulled using media stack API
- llama_text.json => cleaned text used for finetuning the model
- Data_API.py => code to pull data from API
- clean_prep_data.py => code to clean and prepare data for training
- Train_model.py => Train the model
- Eval_model.py => Evaluate the model

Script to run the files

```
python Data_API.py raw_file.json
```

```
python clean_prep_data.py raw_file.json cleaned_text.json
```

```
python Train_model.py cleaned_text.json trained_model_folder
```

```
python Eval_model.py trained_model_folder 
>>> "title of the article"

```

Example runs

```
python Data_API.py output2.json
python clean_prep_data.py output2.json llama_text.json
python Train_model.py llama_text.json llama-7b-trained

python Eval_model.py llama-7b-trained
>>> Ariana Madix Sues Tom Sandoval to Force Sale of Their House

In a shocking turn of events, Ariana Madix has filed a lawsuit against her ex-husband, Tom Sandoval, in order to force the sale of their shared house. According to court documents obtained by E! News, Madix is seeking to have the property sold and the proceeds divided equally between herself and Sandoval.


>>> Oil prices rise for the second time this week

Oil prices surged for a second time this week on Thursday, driven by a combination of factors including tight supply and demand fundamentals, geopolitical tensions, and optimism over global economic growth.

