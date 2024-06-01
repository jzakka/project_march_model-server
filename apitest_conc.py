import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

#제가 테스트할 때 쓴 코드입니다.

payload = ["복수개의 문장을 보낼 수 있는 방법이 없는건가?", 
           "난감한데", 
           "짱개는 혐오가 답이다",
           "코이츠 아주 야발놈인ㅋㅋㅋ",
           "진짜 생각없는 한남새끼들 다 죽어"
           ]*20

def work():
    start = time.time()
    try:
        response = requests.post('http://127.0.0.1:8000/api/classify', json=payload)
        end = time.time()
        return response, end - start
    except requests.RequestException as e:
        end = time.time()
        return e, end - start

with ThreadPoolExecutor(max_workers=10) as pool:
    futures = [pool.submit(work) for _ in range(10)]

for future in as_completed(futures):
    response, duration = future.result()
    print(f"{duration:.5f} sec")
    print("====")
    if isinstance(response, requests.Response):
        print(response.text[:100])  # or response.json() if the response is in JSON format
    else:
        print(f"Request failed: {response}")