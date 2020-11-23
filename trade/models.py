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
  def __init__(self, *args, c0="№", c1="Дата/Время",c2="Цена открытия",c3="Самая высокая цена",c4="Самая низкая цена",c5="Цена закрытия",c6="Ср. цена закрытия",c7="Объем",c8="Наименование",**kwargs):
        super().__init__(*args, **kwargs)
        self.base_columns['date_time'].verbose_name = c1
        self.base_columns['q_open'].verbose_name = c2
        self.base_columns['q_high'].verbose_name = c3
        self.base_columns['q_low'].verbose_name = c4
        self.base_columns['q_close'].verbose_name = c5
        self.base_columns['q_adj_close'].verbose_name = c6
        self.base_columns['q_volume'].verbose_name = c7
        self.base_columns['q_name'].verbose_name = c8
  class Meta:
      model = Quote
      attrs = {'class': 'table table-hover table-sm'
      }