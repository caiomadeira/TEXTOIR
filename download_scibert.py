from transformers import AutoTokenizer, AutoModel
import os

output_dir = "models/scibert_scivocab_uncased"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_name = "allenai/scibert_scivocab_uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)
print('show')