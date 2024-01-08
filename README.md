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

python Data_API.py raw_file.json

python clean_prep_data.py raw_file.json cleaned_text.json

python Train_model.py cleaned_text.json trained_model_folder

python Eval_model.py trained_model_folder 
>>> "title of the article"

