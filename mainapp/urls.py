from django.urls import path,include
from . import views

app_name = 'mainapp'

urlpatterns = [
    path('',views.home,name='home'),
    path('predict/',views.predict,name='predict'),
]