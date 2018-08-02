from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^(\d{1,2})/plus/(\d{1,2})/$', views.add, name='add'),
    url(r'^(\d{1,2})/math/(\d{1,2})/$', views.math, name='math'),
    url(r'^menu$', views.menu, name='menu'),
]