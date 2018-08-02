import random
import os
from xml.etree.ElementTree import fromstring,Element

# -*- coding: utf-8 -*-


def result(price):
    x=float(price)
    result = x-int(float(random.randint(1,10)/100)*float(x))
    return result
"""
def main():
    price="100"
    print(result(price))

if __name__=='__main__':
    main()
    
 """   