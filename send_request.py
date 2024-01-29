import requests

if __name__ == '__main__':
    url = 'http://127.0.0.1:8000/generate'
    user_ask = input("Enter your question: ")
    data = {'prompt': user_ask}
    response = requests.post(url, json=data)
    print(response.json())