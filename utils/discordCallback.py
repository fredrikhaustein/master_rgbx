import requests

def post_to_discord(webhook_url, message):
    data = {"content": message}
    response = requests.post(webhook_url, json=data)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        # You may want to log this error or handle it further
        print(err)