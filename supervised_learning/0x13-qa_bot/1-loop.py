#!/usr/bin/env python3
"""Q/A ChatBot module"""

while True:
    val = input("Q: ")
    exit_list = ['exit', 'quit', 'goodbye', 'bye']
    if val.lower() in exit_list:
        print("A: Goodbye")
        break
    answer = ''
    print("A: {}".format(answer))
