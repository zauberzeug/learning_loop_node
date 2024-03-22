#!/usr/bin/env python3
import requests  # type: ignore

# https://stackoverflow.com/a/39225272/4082686


def g_download(file_id: str, destination: str):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={
        'id': file_id,
        'confirm': 't'},  # large file warning.
        stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


# if __name__ == "__main__":
#     g_download(file_id='1q8nT-CTHt1eZuNjPMbdaavyFnMtDRT-L', destination='test.zip')
