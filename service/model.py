import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# MODEL_PATH = os.getenv("MODEL_PATH", "/root/kc_electra_unsmile")

MODEL_PATH = os.getenv("MODEL_PATH", "/Users/chungsanghwa/projects/final-season/kc_electra_small_unified_with_merged_labels")
#이 코드는 로컬 테스트 용입니다.

class NLPModel:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH,local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only = True)
        self.pipe = TextClassificationPipeline( 
            model = self.model,
            tokenizer = self.tokenizer,
            device=-1,
            return_all_scores=True,
            # function_to_apply='sigmoid'
        ) # 사용을 위한 파이프라인
    
    
    def classify(self, text_content):
        results = self.pipe(text_content, batch_size=4)
        for result in results:
            for ls_pair in result:
                ls_pair['score'] = round(ls_pair['score'], 4)
        return results