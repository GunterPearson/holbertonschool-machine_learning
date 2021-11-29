#!/usr/bin/env python3
"""Pipeline Api"""
import requests


def availableShips(passengerCount):
    """show all starhips avaliable from the API"""
    url = "https://swapi-api.hbtn.io/api/starships"
    r = requests.get(url)
    ship_list = []
    while r.status_code == 200:
        for ship in r.json()["results"]:
            if ship["passengers"] is not None:
                try:
                    num = ship["passengers"].replace(",", "")
                    if int(num) >= passengerCount:
                        ship_list.append(ship["name"])
                except ValueError:
                    pass
        try:
            r = requests.get(r.json()["next"])
        except Exception:
            break
    return ship_list
