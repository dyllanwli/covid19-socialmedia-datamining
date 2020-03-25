# import essential packages
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
import os

# config 
# folderPath = "/scratch/group/gis-cidi/data-geo-only"
# outputPath = "./twitter-data-geo-output"

# all-data-config 
folderPath = "/scratch/group/gis-cidi/twitter-data"
outputPath = "./twitter-data-output"

def readAllTwitterCSV(filePath):
    with open(filePath, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='"')
        yield next(datareader)
        
        for row in datareader:
            rowData = [x for x in row]
            _json = rowData[1]
            yield eval(_json)
        return
    
    
    
def main():
    for file in os.listdir(folderPath)[::-1]:
        filePath = folderPath + "/" + file
        outputFilePath = "{}/{}.json".format(outputPath, file.split(".")[0])
        if os.path.isfile(outputFilePath):
            return
        print("processing {}".format(file))
        output = []
        allRow = readAllTwitterCSV(filePath)
        for index, i in enumerate(allRow):
            if index > 0:
                output.append(i)
        with open(outputFilePath, "w") as outputFile:
            json.dump(output, outputFile)
            
            
main()