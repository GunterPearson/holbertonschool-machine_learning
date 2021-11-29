#!/usr/bin/env python3
"""Pipeline Api"""
import requests
import sys
from datetime import datetime


if __name__ == '__main__':
    """pipeline api"""
    url = sys.argv[1]
    response = requests.get(url)

    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        string = 'X-Ratelimit-Reset'
        date = datetime.fromtimestamp(int(response.headers[string]))
        min = str((date - datetime.now())).split(':')[1]
        min = int(min)
        print("Reset in {} min".format(min))
    else:
        print(response.json()["location"])
