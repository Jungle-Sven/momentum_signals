import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression

from datetime import datetime
import redis

import json

# Connect to the Redis server
redis_host = 'momentum_redis_container'  # Change this to the Redis server address if running on a different machine
redis_port = 6379         # Default Redis port
redis_db = 0              # Redis database number
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db)

#determines how precise is the model
QUANTILE = 0.9

# Custom serialization function for datetime objects
def datetime_serializer(obj):
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')  # Customize the format here
    raise TypeError("Type not serializable")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class Momentum:
    '''read
    resample
    save
    calc threshold
    calc momentum
    return result'''
    def __init__(self):
        self.symbols = ['BTC-USD', 'ETH-USD']
        self.signals = {'BTC-USD': {}, 'ETH-USD': {}}
        
    def read_data_ob(self, symbol):
        ob_lname = 'ob_' + symbol
        json_data_list = redis_client.lrange(ob_lname, 0, -1)
        return [json.loads(json_data) for json_data in json_data_list]
    
    def read_data_trade(self, symbol):
        trade_lname = 'trade_' + symbol
        json_data_list = redis_client.lrange(trade_lname, 0, -1)
        return [json.loads(json_data) for json_data in json_data_list]
    
    def convert_ob_data(self, df):
        numeric_fields = ['best_bid_price', 'best_bid_size', 'best_ask_price', 'best_ask_size']

        # Convert numeric fields to numeric types
        df[numeric_fields] = df[numeric_fields].apply(pd.to_numeric, errors='coerce')

        return df
    
    def convert_trade_data(self, df):
        numeric_fields = ['amount', 'price']
        
        # Convert numeric fields to numeric types
        df[numeric_fields] = df[numeric_fields].apply(pd.to_numeric, errors='coerce')

        return df
    
    def create_trades_df(self, data):
        numeric_fields = ['amount', 'price']
        # Create the DataFrame using pandas.DataFrame.from_records() with the converters parameter
        
        #timestamp, side, amount, price
        df = pd.DataFrame(data, columns = ['receipt_timestamp', \
            'trade_timestamp', 'symbol', 'side', 'amount', 'price'])
        df.dropna(inplace=True)
        df['receipt_timestamp'] = pd.to_datetime(df['receipt_timestamp'], unit='s')
        df.set_index('receipt_timestamp', inplace=True)
        #convert Decimal to float, fuck this shit
        df['amount'] = df['amount'].astype(float)
        return df
    
    def create_ob_df(self, data):
        numeric_fields = ['best_bid_price', 'best_bid_size', 'best_ask_price', 'best_ask_size']
        
        #timestamp, side, amount, price
        df = pd.DataFrame(data, columns = ['receipt_timestamp', 'symbol', \
            'best_bid_price', 'best_bid_size', 'best_ask_price', 'best_ask_size'])
        df.dropna(inplace=True)
        df['timestamp'] = pd.to_datetime(df['receipt_timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        return df

    def buy_sell_volume(self, df):
        df['buy_volume'] = df['amount'].where(df['side'] == 'buy', 0)
        df['sell_volume'] = df['amount'].where(df['side'] == 'sell', 0)
        return df
    
    def resample_trades(self, df, timeframe):
        df = df.resample(timeframe).agg({'amount': ['count', 'sum'], 'price': ['mean', 'first', 'max', 'min', 'last'], 'buy_volume':'sum', 'sell_volume':'sum'})
        return df
    
    #this is executed trades quantity, not imbalance
    def n_executed_trades_signal(self, df):
        df['n_trades'] = df[('amount', 'count')].rolling(5).mean()
        executed_trades_threshold = df['n_trades'].quantile(QUANTILE)
        #print(f'executed_trades_threshold is {executed_trades_threshold}')
        df['n_executed_trades_signal'] = 0
        df['n_executed_trades_signal'] = df['n_trades'].apply(lambda x: 1 if x >executed_trades_threshold else 0)
        return df
    
    def calc_high_volume(self, df):
        vol_threshold = df[('amount', 'sum')].quantile(QUANTILE)
        #print(f'vol_threshold is {vol_threshold}')
        df['high_volume'] = df[('amount', 'sum')].apply(lambda x: 1 if x >= vol_threshold else 0)
        return df
    
    def high_volume_signal(self, df):
        df['high_volume'] = df['high_volume'].rolling(5).mean()
        high_vol_threshold = df['high_volume'].quantile(QUANTILE)
        #print(f'high_vol_threshold is {high_vol_threshold}')
        df['volume_signal'] = df['high_volume'].apply(lambda x: 1 if x >high_vol_threshold else 0)
        return df

    def buy_sell_imbalance(self, df):
        df['total_amount'] = df[('amount', 'sum')].rolling(5).mean()
        buy_sell_amount_threshold = df['total_amount'].quantile(QUANTILE)
        #print(f'buy_sell_amount_threshold is {buy_sell_amount_threshold}')
        df['buy_sell_imbalance'] = 0
        df.loc[df[('buy_volume', 'sum')] > buy_sell_amount_threshold, 'buy_sell_imbalance'] = 1
        df.loc[df[('sell_volume', 'sum')] > buy_sell_amount_threshold, 'buy_sell_imbalance'] = -1
        return df
    
    def buy_sell_imbalance_signal(self, df):
        df[('buy_sell_imbalance', '')] = df[('buy_sell_imbalance', '')].rolling(10).mean()
        buy_sell_imbalance_threshold =df[('buy_sell_imbalance', '')].quantile(QUANTILE)
        #print(f'buy_sell_imbalance_threshold is {buy_sell_imbalance_threshold}')
        df['buy_sell_imbalance_signal'] = df.apply(lambda x: 1 if x[('buy_sell_imbalance', '')] > buy_sell_imbalance_threshold else (-1 if x[('buy_sell_imbalance', '')] < -buy_sell_imbalance_threshold else 0), axis=1)

        return df
    
    #OB
    def calc_mid_price(self, df):
        df['mid_price'] = (df['best_ask_price'] + df['best_bid_price']) / 2
        return df

    def resample_orderbook_updates(self, df, timeframe='10S'):
        df = df.resample(timeframe).agg({'symbol':'count', 'best_ask_size':'sum', \
            'best_bid_size':'sum', 'mid_price':'mean'})
        #df = df.fillna(0)
        return df
    
    def n_ob_updates_signal(self, df):
        df['amount'] = df['symbol'].rolling(5).mean()
        ob_updates_threshold = df['amount'].quantile(QUANTILE)
        #print(f'ob_updates_threshold is {ob_updates_threshold}')
        df['n_ob_updates_signal'] = 0
        df['n_ob_updates_signal'] = df['amount'].apply(lambda x: 1 if x >ob_updates_threshold else 0)
        return df
    
    def calc_momentum_signal(self, df):
        df['momentum_signal'] = df[('n_executed_trades_signal', '')] + \
            df['n_ob_updates_signal'] + df[('volume_signal', '')]
        return df
    
    def momentum_signal_with_direction(self, df):
        df['signal_with_direction'] = 0  
        df.loc[(df['momentum_signal'] >= 2) & (df[('buy_sell_imbalance_signal', '')] == -1), 'signal_with_direction'] = -1
        df.loc[(df['momentum_signal'] >= 2) & (df[('buy_sell_imbalance_signal', '')] == 1), 'signal_with_direction'] = 1
        return df
    
    '''linear regression'''
    def calc_mas(self, df):
        df['ma10'] = df[('price', 'mean')].rolling(10).mean()
        df['ma50'] = df[('price', 'mean')].rolling(50).mean()
        df['ma100'] = df[('price', 'mean')].rolling(100).mean()
        #df = df[100:] #remove NaN
        return df
    
    def create_train_test_set(self, df):
        df = df.fillna(0)
        features = df[['ma10', 'ma50', 'ma100', \
            'momentum_signal', 'signal_with_direction']]
        target = df[('price', 'mean')]
        return features, target
    
    def train(self, features, target):
        lr = LinearRegression()
        lr.fit(features, target)
        return lr
        
    def predict(self, lr, features):
        result = lr.predict(features)
        return result
    
    def create_change_df(self, change, change_norm):
        # Create a DataFrame with the normalized prediction and change columns
        lr_df = pd.DataFrame({
                'change': change, \
                'change_norm': change_norm})
        return lr_df
    
    def calc_change_signal(self, df):
        threshold = df['change_norm'].quantile(QUANTILE)
        #print('change_norm threshold', threshold)
        df['change_signal'] = 0
        df.loc[df['change_norm'] > threshold, 'change_signal'] = 1
        df.loc[df['change_norm'] < -threshold, 'change_signal'] = -1
        return df
    
    def gather_data(self, symbol, combined_df, lr_df):
        #momentum_signal: 0-3 where 0 is no trend, 3 = market trending
        #signal_with_direction can be -1 to +1, sell and buy
        #price_change is predicted price change in $
        #price_change_norm is normalized predicted price change
        #change_signal -1 to 1 on serious price change prediction
        self.signals[symbol] = {
            'symbol': symbol, \
            'timestamp': datetime_serializer(datetime.utcnow()), \
            'price': round(combined_df[('price', 'mean')].iloc[-1], 4), \
            'momentum_signal': combined_df['momentum_signal'].iloc[-2], \
            'signal_with_direction': combined_df['signal_with_direction'].iloc[-2], \
            'change_signal': lr_df['change_signal'].iloc[-1], \
            'trades_receipt_timestamp': self.trades_update_timestamp, \
            'ob_receipt_timestamp': self.ob_update_timestamp}
        
    def save_data(self, symbol):
        # Save data to Redis
        data = self.signals[symbol]
        print((data['symbol']), (data['timestamp']), (data['price']), \
              (data['momentum_signal']), (data['signal_with_direction']), \
                (data['change_signal']), data['trades_receipt_timestamp'], \
                data['ob_receipt_timestamp'])
        data = json.dumps(data, cls=NpEncoder)
        #json.dumps(data, cls=NpEncoder)
        key_name = 'signal_' + symbol                
        redis_client.set(key_name, data)
   
    def run(self):
        while True:
            time.sleep(1)

            for symbol in self.symbols:
                data_start_time = time.time()
                
                ob_data = self.read_data_ob(symbol)
                trade_data = self.read_data_trade(symbol)

                trades_df = self.create_trades_df(trade_data)
                ob_df = self.create_ob_df(ob_data)

                trades_df = self.convert_trade_data(trades_df)
                ob_df = self.convert_ob_data(ob_df)

                self.trades_update_timestamp = trades_df['trade_timestamp'].iloc[-1]
                print(ob_df.columns)
                self.ob_update_timestamp = ob_df['receipt_timestamp'].iloc[-1]
                
                data_finish_time = time.time()
                print(f'data time: {data_finish_time-data_start_time}')
            
                #working with trades
                trades_start_time = time.time()
                trades_df = self.buy_sell_volume(trades_df)
                trades_df = self.resample_trades(trades_df, timeframe='10S')
                trades_df = self.n_executed_trades_signal(trades_df)
                trades_df = self.calc_high_volume(trades_df)
                trades_df = self.high_volume_signal(trades_df)

                trades_df = self.buy_sell_imbalance(trades_df)
                trades_df = self.buy_sell_imbalance_signal(trades_df)

                trades_finish_time = time.time()
                print(f'trades time: {trades_finish_time-trades_start_time}')
            
                ob_start_time = time.time()
                #working with OB 
                ob_df = self.calc_mid_price(ob_df)
                ob_df = self.resample_orderbook_updates(ob_df, timeframe='10S')
                ob_df = self.n_ob_updates_signal(ob_df)

                ob_finish_time = time.time()
                print(f'ob time: {ob_finish_time-ob_start_time}')
            
                combined_start_time = time.time()

                #combining dfs
                combined_df = pd.concat([trades_df, ob_df], axis=1)
                combined_df = combined_df.fillna(method='ffill')
                
                combined_df = self.calc_momentum_signal(combined_df)

                combined_df = self.momentum_signal_with_direction(combined_df)

                combined_finish_time = time.time()
                print(f'combined time: {combined_finish_time-combined_start_time}')

                linreg_start_time = time.time()

                #lin reg
                combined_df = self.calc_mas(combined_df)
            
                #print(combined_df)

                features, target = self.create_train_test_set(combined_df)
                lr = self.train(features, target)
                prediction = self.predict(lr, features)

                # Normalize the prediction data
                normalized_prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
                # Calculate the change compared to the previous element
                change = np.diff(prediction)
                change_norm = np.diff(normalized_prediction)

                lr_df = self.create_change_df(change, change_norm)
                lr_df = self.calc_change_signal(lr_df)

                linreg_finish_time = time.time()
                print(f'linreg time: {linreg_finish_time-linreg_start_time}')            

                #gather data into one place
                self.gather_data(symbol, combined_df, lr_df)

                #print(f'now: {datetime.utcnow()}')

                self.save_data(symbol)

if __name__ == '__main__':
    signals = Momentum()
    signals.run()