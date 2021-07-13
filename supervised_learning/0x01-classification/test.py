#!/usr/bin/env python3
""" test """


def save(filename):
    """ save neural network"""
    if type(filename) is not str:
        return
    if filename[-4:] != ".pkl":
        filename = filename + ".pkl"
    print(filename)

string = "first"
save(string)
    
