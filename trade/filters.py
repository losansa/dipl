import django_filters as df
from .models import Quote, QuotesName
from django_filters.widgets import RangeWidget
class QuoteFilter(df.FilterSet):
  date_time = df.DateFromToRangeFilter(widget=RangeWidget(attrs={'type': 'date'}))
  class Meta:
    model = Quote
    fields = ['q_name','date_time']

