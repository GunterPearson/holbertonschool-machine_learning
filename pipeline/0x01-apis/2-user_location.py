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
        date = datetime.fromtimestamp(int(response.headers['X-Ratelimit-Reset']))
        min = str((date - datetime.now())).split(':')[1]
        print("Reset in {} minutes".format(min))
    else:
        print(response.json()["location"])
