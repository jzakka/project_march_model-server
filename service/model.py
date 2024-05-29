import os
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

#MODEL_PATH = os.getenv("MODEL_PATH", "/root/kc_electra_unsmile")

#MODEL_PATH = os.getenv("MODEL_PATH", "/Users/chungsanghwa/projects/final-season/kc_electra_small_unified_with_merged_labels")
MODEL_PATH = os.getenv("MODEL_PATH","C:/Users/허대현/tiny_git/NLPService/project_march_model-server/kc_electra_small_unified")
#이 코드는 로컬 테스트 용입니다.

class NLPModel:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH,local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only = True, max_length=512,truncate=True)
        self.pipe = TextClassificationPipeline( 
            model = self.model,
            tokenizer = self.tokenizer,
            device=-1,
            top_k=None,
            function_to_apply='sigmoid'
        ) # 사용을 위한 파이프라인
    
    
    async def classify(self, text_content):
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: self.pipe(text_content, batch_size=4))
        for result in results:
            for ls_pair in result:
                ls_pair['score'] = round(ls_pair['score'], 4)
        return results
    

    def classify_worker(self, texts, output_list):
        results = self.classify(texts)
        output_list.extend(results)

    def classify_mt(self, text_content, num_workers=4): # 멀티 쓰레드 적용
        if len(text_content) < num_workers: 
            num_workers = len(text_content)
        chunk_size = len(text_content) // num_workers
        chunks = [text_content[i:i + chunk_size] for i in range(0, len(text_content), chunk_size)]

        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for chunk in chunks:
                executor.submit(self.classify_worker, chunk, results)
            #future_to_chunk = {executor.submit(self.classify_worker, chunk, results): chunk for chunk in chunks}
            #for future in as_completed(future_to_chunk):
                #future.result()  # 예외 처리를 위해 필요할 수 있음

        return results
