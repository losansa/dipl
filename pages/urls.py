from django.urls import path, include
from . import views

urlpatterns = [
    path('',views.main_view, name=''),
    # path('',views.my_second_view, name=''),
    # path('test_form/',views.test_form,name='test_form'),
    # path('contact/',views.contact, name='contact'),
    # path('graph/',views.graph, name='graph'),
]
