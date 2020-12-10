#!/usr/bin/env python3
""" 17-integrate"""


def poly_integral(poly, C=0):
    """ integral math"""
    if type(poly) != list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    poly.insert(0, 0)
    for x in range(1, len(poly)):
        poly[x] = poly[x] / x
    return poly
