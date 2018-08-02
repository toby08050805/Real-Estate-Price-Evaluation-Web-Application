from django.test import TestCase
import csv
import os

# Create your tests here.

currentFile = 'example_input.csv' 
with open(currentFile, "r") as myCSV:
    myReader = csv.reader(myCSV)
    row1= next(myReader)
    
print( row1)


        