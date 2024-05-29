import requests
import json

# POST 요청에 보낼 데이터
payload = ["복수개의 문장을 보낼 수 있는 방법이 없는건가?", "난감한데", "짱개는 혐오가 답이다","코이츠 아주 야발놈인ㅋㅋㅋ","진짜 생각없는 한남새끼들 다 죽어"]*100

# POST 요청 보내기
response = requests.post('http://127.0.0.1:8000/api/classify', json=payload)
# 응답 상태 코드 확인
print(response.status_code)
# 응답 데이터 출력 (JSON 형식일 경우)
if response.headers['Content-Type'] == 'application/json':
    data = response.json()
    print(data)
else:
    print(response.text)


# POST 요청 보내기
response = requests.post('http://127.0.0.1:8000/api/classify_mt', json=payload)
# 응답 상태 코드 확인
print(response.status_code)
# 응답 데이터 출력 (JSON 형식일 경우)
if response.headers['Content-Type'] == 'application/json':
    data = response.json()
    print(data)
else:
    print(response.text)