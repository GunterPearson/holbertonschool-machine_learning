#!/usr/bin/env python3
"""Pipeline Api"""
import requests


def sentientPlanets():
    """returns the list of names of the home planets of all sentient species"""
    url = "https://swapi-api.hbtn.io/api/species"
    r = requests.get(url)
    world_list = []
    while r.status_code == 200:
        for species in r.json()["results"]:
            url = species["homeworld"]
            if url is not None:
                ur = requests.get(url)
                world_list.append(ur.json()["name"])
        try:
            r = requests.get(r.json()["next"])
        except Exception:
            break
    return world_list
