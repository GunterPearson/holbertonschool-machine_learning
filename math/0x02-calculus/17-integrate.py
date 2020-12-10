#!/usr/bin/env python3
""" 17-integrate"""


def poly_integral(poly, C=0):
    """ integral math"""
    if type(poly) != list or len(poly) < 1:
        return None
    if type(C) != int and type(C) != float:
        return None
    result = all(type(s) in [int, float] for s in poly)
    if not result:
        return None
    poly.insert(0, 0)
    for x in range(1, len(poly)):
        poly[x] = poly[x] / x
        poly[x] = poly[x] + C
    for y in range(len(poly)):
        if poly[y] % 1 == 0:
            poly[y] = int(poly[y])
    return poly
