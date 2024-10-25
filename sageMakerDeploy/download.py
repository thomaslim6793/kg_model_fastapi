from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained('Babelscape/rebel-large')
tokenizer = AutoTokenizer.from_pretrained('Babelscape/rebel-large')

model.save_pretrained('model')
tokenizer.save_pretrained('model')