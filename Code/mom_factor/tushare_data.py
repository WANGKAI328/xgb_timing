import tushare as ts
import pandas as pd
import numpy as np
import Code.utils.MLAM as mlam
pro = ts.pro_api('77cd7df269ff0afca67ffc4a7d0d84c2d6afae26d047472d1b9b637d')



if __name__ == "__main__":
    hs300_index = pro.index_daily(**{
        "ts_code": "000300.SH",
        "trade_date": "",
        "start_date": 20100101,
        "end_date": 20250304,
        "limit": "",
        "offset": ""}, 
        fields=["trade_date","close","open","high","low","pre_close","change","pct_chg","vol","amount"])
    hs300_index.rename(columns = {"trade_date":"trade_day"})
    hs300_index = hs300_index.sort_values(by = 'trade_date')
    hs300_index.to_csv("hs300_index.csv",index = False)

    


    
        