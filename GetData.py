import tushare as ts
import json

def daily_basic(start,end,filename,stock_code='',date=''):
	'''抓取单日指标'''
	
	#抓数据
	pro = ts.pro_api()
	df = pro.daily_basic(ts_code=stock_code, trade_date=date, start_date=start,end_date=end,fields='trade_date,pe,pb,total_share,total_mv')
	
	#数据按日期升序排列
	df=df.sort_values(by = 'trade_date')
	#转成字典
	daily_basic={col:df[col].tolist() for col in df.columns}

	with open(filename,'w') as f_obj:
		json.dump(daily_basic,f_obj)


def daily_one(stock_code,start,end,filename):
	'''抓取单日行情'''
	
	pro = ts.pro_api()
	df = pro.daily(ts_code=stock_code,start_date=start,end_date=end)
	
	df=df.sort_values(by = 'trade_date')
	daily={col:df[col].tolist() for col in df.columns}
	
	with open(filename,'w') as f_obj:
		json.dump(daily,f_obj)
if __name__ == '__main__':
	daily_one(start='20170101',end='20190215',stock_code='000002.SZ',filename='daily')
	daily_basic(start='20170101',end='20190215',stock_code='000002.SZ',filename='daily_basic')
