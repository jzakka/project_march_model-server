import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForSequenceClassification, TextClassificationPipeline

MODEL_PATH=os.getenv("MODEL_PATH", "/root/kc_electra_unsmile")

class NLPModel:
    def __init__(self):
        self.model_modified = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH,local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only = True)
        self.pipe = TextClassificationPipeline( 
            model = self.model_modified,
            tokenizer = self.tokenizer,
            return_all_scores=True,
        ) # 사용을 위한 파이프라인
    
    def classify(self, text_content):
        return self.pipe(text_content)[0]