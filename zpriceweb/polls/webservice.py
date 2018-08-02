import requests
import os
import csv
from xml.etree.ElementTree import fromstring,Element

# -*- coding: utf-8 -*-



def webservice(addressline,zipcode):
#1540 Oakleaf Ave, Healdsburg, CA 95448

    zwsid ="X1-ZWz190yqu7tvrf_5qhl2"   # the ID I register in https://www.zillow.com/howto/api/APIOverview.htm
    #address="10297 Latney Rd, Fairfax, VA 22032"
    address=addressline
    #citystatezip= "95448"
    citystatezip=zipcode

    print(addressline)
    print(zipcode)

    response = requests.get("http://www.zillow.com/webservice/GetSearchResults.htm?zws-id=%s&address=%s&citystatezip=%s"%(zwsid,address,citystatezip)) #put id,address,and zipcode to requrire imformation
    print("http://www.zillow.com/webservice/GetSearchResults.htm?zws-id=%s&address=%s&citystatezip=%s"%(zwsid,address,citystatezip))
    root = fromstring(response.content) #defind the root of XML tree


    tmptag=root.find("response") #looking amount data in XML tree
    if(root.find("response") is not None):
        
        tmptag=tmptag.find("results")
        tmptag=tmptag.find("result")
        tmptag=tmptag.find("zestimate")
        tmptag=tmptag.find("amount")
    else:
        return "Can't find this address on database"
        #print(tmptag.text)
    price =tmptag.text
    
    tmptag=root.find("response")
    if(root.find("response") is not None):
        
        tmptag=tmptag.find("results")
        tmptag=tmptag.find("result")
        tmptag=tmptag.find("zpid")
    else:
        return "Can't find this address on database"
    zpid=tmptag.text
    
    tmptag=root.find("response")
    if(root.find("response") is not None):
        tmptag=tmptag.find("results")
        tmptag=tmptag.find("result")
        tmptag=tmptag.find("address")
        tmptag=tmptag.find("latitude")
    else:
        return "Can't find this address on database"
    lat=tmptag.text
    
    tmptag=root.find("response")
    if(root.find("response") is not None):
        tmptag=tmptag.find("results")
        tmptag=tmptag.find("result")
        tmptag=tmptag.find("address")
        tmptag=tmptag.find("longitude")
    else:
        return "Can't find this address on database"
    long=tmptag.text
    
    response = requests.get("http://www.zillow.com/webservice/GetDeepComps.htm?zws-id=%s&zpid=%s&count=1"%(zwsid,zpid))
    print("http://www.zillow.com/webservice/GetDeepComps.htm?zws-id=%s&zpid=%s&count=1"%(zwsid,zpid))
    root = fromstring(response.content)
    tmptag=root.find("response")
    if(root.find("response") is not None):
        tmptag=tmptag.find("properties")
        tmptag=tmptag.find("principal")
        tmptag=tmptag.find("taxAssessment")
    taxamount=tmptag.text
    #print(taxamount)
    root = fromstring(response.content)
    tmptag=root.find("response")
    if(root.find("response") is not None):
        tmptag=tmptag.find("properties")
        tmptag=tmptag.find("principal")
        tmptag=tmptag.find("finishedSqFt")
    finishedSqFt=tmptag.text
    print(finishedSqFt)
    root = fromstring(response.content)
    tmptag=root.find("response")
    if(root.find("response") is not None):
        tmptag=tmptag.find("properties")
        tmptag=tmptag.find("principal")
        tmptag=tmptag.find("lotSizeSqFt")
    lotSizeSqFt=tmptag.text
    print(lotSizeSqFt)
    
    root = fromstring(response.content)
    tmptag=root.find("response")
    if(root.find("response") is not None):
        tmptag=tmptag.find("properties")
        tmptag=tmptag.find("principal")
        tmptag=tmptag.find("finishedSqFt")
    finishedSqFt=tmptag.text
    print(finishedSqFt)
    
    root = fromstring(response.content)
    tmptag=root.find("response")
    if(root.find("response") is not None):
        tmptag=tmptag.find("properties")
        tmptag=tmptag.find("principal")
        tmptag=tmptag.find("yearBuilt")
    yearBuilt=tmptag.text
    yearBuilt=tmptag.text
    print(yearBuilt)
    
    currentFile = 'example_input.csv' 
    with open(currentFile, "r") as myCSV:
        myReader = csv.reader(myCSV)
        row1= next(myReader)
    lat=int(float(lat)*1000000)
    long=int(float(long)*1000000)
    taxamount=float(taxamount)
    finishedSqFt=int(finishedSqFt)
    lotSizeSqFt=int(lotSizeSqFt)
    finishedSqFt=int(finishedSqFt)
    yearBuilt=int(yearBuilt)
    listdata=taxamount,finishedSqFt,lat,lotSizeSqFt,finishedSqFt,long,yearBuilt
    print(row1)
    print(listdata)

    outfreefile = 'predictInput.csv' 
    with open('outfile.csv', 'w',newline='') as csvfile:
        csvCursor = csv.writer(csvfile)
        csvCursor.writerow(row1) 
        csvCursor.writerow(listdata)            

    
    
    return tmptag.text

 


#print(webservice("10297 Latney Rd, Fairfax, VA 22032"))





#print(lowCurrency)

def main():
    print(webservice("316 Mountain View Dr, Healdsburg, CA 95448","95448"))
    
if __name__=='__main__':
    main()
    
    