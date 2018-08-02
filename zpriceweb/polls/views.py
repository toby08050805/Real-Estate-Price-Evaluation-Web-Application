from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django import template
from django.template.loader import get_template
from django.shortcuts import render_to_response
from polls.webservice import webservice
from polls.result import result




def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def add(request, a, b):
    s = int(a)+int(b)
    return HttpResponse(str(s))



def math(request, a, b):
    a = int(a)
    b = int(b)
    s = a + b
    d = a - b
    p = a * b
    q = a / b
    t = get_template('math.html')
    return render_to_response('math.html',{'s':s, 'd':d, 'p':p, 'q':q})

def menu(request):
    food1 = { 'name':'111', 'price':60, 'comment':'yummy', 'is_spicy':False }
    food2 = { 'name':'2222', 'price':100, 'comment':'fation', 'is_spicy':True }
    foods = [food1,food2]
    return render_to_response('menu.html',locals())
def welcome(request):
    if 'address' in request.GET:
        print("I am here")
        zprice=webservice(request.GET['address'],'22032')
        return render_to_response('InputForm.html',locals())#test
        return HttpResponse('The price is:'+zprice)
        #webservice={'price':zprice}
    else:
        return render_to_response('welcome.html',locals())
def InputForm(request):
    test={}
    
    if 'mailing-address-1' in request.GET:
        address=request.GET['mailing-address-1']+', '+request.GET['city']+', '+request.GET['state']+' '+request.GET['zip']
        zipcode=request.GET['zip']
        zprice=webservice(address,zipcode)
        if(zprice=="Can't find this address on database"):
            test['address']="The address you enter is: "+address+"  "+zprice
            return render_to_response('InputForm2.html',locals())
        test['ziprice']="The ziprice is: "+"$"+str(float(zprice))
        test['address']="The address you enter is: "+address
        test['result']="Our estimate price is: "+"$"+str(result(float(zprice)))
        #return HttpResponse('The price is:'+zprice)
        return render_to_response('InputForm2.html',locals())
        #return HttpResponse('The price is:'+" "+address)
    else:
        
        return render_to_response('InputForm2.html',locals())

