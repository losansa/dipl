from django.shortcuts import render
from .models import Quote, QuoteTable
from .filters import QuoteFilter
from django_tables2 import RequestConfig
from django.core.cache import cache

from django_filters.views import FilterView
from django_tables2.views import SingleTableMixin
class FilteredQuoteListView(SingleTableMixin, FilterView):
    table_class = QuoteTable
    model = Quote
    template_name = "trade/quotes_table.html"
    filterset_class = QuoteFilter 