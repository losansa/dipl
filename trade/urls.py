from .views import  FilteredQuoteListView
from django.urls import re_path, include

urlpatterns = [
    re_path(r'', FilteredQuoteListView.as_view(), name='table_with_quotes'),
]
