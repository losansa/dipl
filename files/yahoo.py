import yfinance as yf

data = yf.download('AAPL','2016-01-01','2019-08-01')

import matplotlib.pyplot as plt


data['Adj Close'].plot()
# plt.savefig('books_read.png')
plt.show()

import pandas as pd
tickers_list = ['AAPL', 'WMT', 'IBM', 'MU', 'BA', 'AXP']

data = pd.DataFrame(columns=tickers_list)


for ticker in tickers_list:
     data[ticker] = yf.download(ticker,'2016-01-01','2019-08-01')['Adj Close']

data.head()

# data.to_csv('out.csv', index=False)  

# Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
  # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
from sqlalchemy import create_engine, MetaData, Table, Integer, String, Column
engine = create_engine('postgresql://lunalo108:108Angel@localhost:5432/myproject')

# from .models import Quotes_Name
# for quotes_name in Quotes_Name:
# data = yf.download(tickers=quotes_name, period="2d", interval="1mo")
tickers_list = ['MOEXBC.ME', 'IMOEX.ME', 'RVI.ME', 'MOEX10.ME', 'RTSI.ME', 'RGBI.ME']
data = yf.download(tickers="IMOEX.ME", period="1mo", interval="5m")
# meta = MetaData()
print(data.tail())

# quotes = Table('trade_quotes', meta,
#               Column('id', Integer, primary_key=True),
#               Column('date_time', DateTimeField(auto_now_add=True)),
#               Column('q_open', FloatField(max_length=120)),
#               Column('q_high', FloatField(max_length=120)),
#               Column('q_low', FloatField(max_length=120)),
#               Column('q_close', FloatField(max_length=120)),
#               Column('q_adj_close', FloatField(max_length=120)),
#               Column('q_volume', FloatField(max_length=120))
#               )
data = data.reset_index()
data.rename(columns={'Open': 'q_open', 'High': 'q_high', 'Low': 'q_low', 'Close': 'q_close', 'Adj Close': 'q_adj_close', 'Volume': 'q_volume','Datetime': 'date_time'}, inplace=True)
data['q_name_id'] = 2
# print(data.columns.tolist())
data['date_time'] = data['date_time'].map(lambda x: str(x)[:19]) 
print(data.tail())
# data.to_csv('fdsfs.csv', index=True)
data.to_sql('trade_quote', con=engine, if_exists='append', index=False)
# Print the data
# print(data.tail())

# from pandas_datareader import data as pdr

# import yfinance as yf
# yf.pdr_override() # <== that's all it takes :-)

# # download dataframe
# data = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30")