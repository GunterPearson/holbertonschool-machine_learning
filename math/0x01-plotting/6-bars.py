#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
names = ['Farrah', 'Fred', 'Felicia']
f_names = {"apples": 'red', "bananas": 'yellow',
           "oranges": '#ff8000', "peaches": '#ffe5b4'}

i = 0
for k, v in f_names.items():
    bottom = 0
    for x in range(i):
        bottom += fruit[x]
    plt.bar(names, fruit[i], width=0.5, color=v,
            align='center', label=k, bottom=bottom)
    i += 1

plt.title("Number of Fruit per Person")
plt.ylim(0, 80)
plt.ylabel("Quantity of Fruit")
plt.legend()
plt.show()
