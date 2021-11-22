#!/usr/bin/env python3
"""from numpy to pandas"""
import pandas as pd


dic = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
    }

df = pd.DataFrame(dic, index=list("ABCD"))
