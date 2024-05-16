from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForSequenceClassification, TextClassificationPipeline


class NLPModel:
    def __init__(self):
        self.model_modified = AutoModelForSequenceClassification.from_pretrained("/Users/chungsanghwa/projects/final-season/kc_electra_unsmile",local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained("/Users/chungsanghwa/projects/final-season/kc_electra_unsmile", local_files_only = True)
        self.pipe = TextClassificationPipeline( 
            model = self.model_modified,
            tokenizer = self.tokenizer,
            return_all_scores=True,
        ) # 사용을 위한 파이프라인
    
    def classify(self, text_content):
        return self.pipe(text_content)[0]