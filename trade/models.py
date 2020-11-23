from django.db import models

class QuotesName(models.Model):
  name = models.TextField()
  def __str__(self):
    return self.name
 
class Quote(models.Model):
  date_time = models.DateTimeField(auto_now_add=True)
  q_open = models.FloatField(max_length=120)
  q_high = models.FloatField(max_length=120)
  q_low = models.FloatField(max_length=120)
  q_close = models.FloatField(max_length=120)
  q_adj_close = models.FloatField(max_length=120)
  q_volume = models.FloatField(max_length=120)
  q_name=models.ForeignKey(
    QuotesName,
    on_delete=models.CASCADE,
    default='1'
  )
 
import django_tables2 as tables
class QuoteTable(tables.Table):
    class Meta:
        model = Quote
        attrs = {'class': 'table table-hover table-sm'
        }