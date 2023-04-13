import pandas as pd 

def preprocessor_wind(df, max_capacity):
    df = df.clip(0,24)**3
    df = df/df.max()*max_capacity
    df.rename("generation", inplace = True)
    return df

def preprocessor_solar(df, max_capacity):
    df = df/df.max()*max_capacity
    df.rename("generation", inplace = True)
    return df

def preprocessor_consumption(df, max_capacity):
    df = df/df.max()*max_capacity
    df.rename("consumption", inplace=True)
    return df

def preprocessor_spot_price(df):
    df.rename("spot_price", inplace=True)
    return df/1000

def preprocessor_fcr_price(df):
    assert len(df) == 2190 
    time =pd.date_range(periods=2190,freq ="4H", start="2019-01-01 00:00:00")
    df = pd.DataFrame(df.values, index=time)
    df = df.resample("1H", closed ="right").mean().fillna(method="ffill")
    df.columns = ["fcr_price"]
    return df.reset_index(drop = True)/4000


if __name__=="__main__":
    print(preprocessor_wind)