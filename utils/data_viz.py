import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 


def plot_input_data( folder_path = "data\consumption" ):
    
    
    fig, ax = plt.subplots(2,2, figsize=(10,10) )
    for file in os.listdir(folder_path):
        df = pd.read_csv(folder_path + "/" + file)
        df["date"] = pd.date_range(start="1/1/2020", periods=len(df), freq="H")
        df.set_index("date", inplace=True, drop=True)
        data_mean = df.groupby(lambda x : x.hour).mean()
        data_std = df.groupby(lambda x : x.hour).std()
        p = ax[0,0].plot( data_mean.index,data_mean["Power [kW]"], label=file)
        ax[0,0].fill_between(data_mean.index,data_mean["Power [kW]"] - data_std["Power [kW]"], data_mean["Power [kW]"] +  data_std["Power [kW]"], alpha=0.1)
    ax[0,0].set_xlabel("Time of day")
    ax[0,0].set_ylabel("Power [kW]")
    ax[0,0].set_title("Consumption profile ")
    ax[0,0].legend()

    df = pd.read_csv("data/fcr_price/random_fcr.csv")

    df["date"] =pd.date_range(periods=2190,freq ="4H", start="2019-01-01 00:00:00")
    df.set_index("date", inplace=True, drop=True)
    df = df.resample("1H", closed ="right").mean().fillna(method="ffill")/4000
    df.columns = ["fcr_price"]
    data_mean = df.groupby(lambda x : x.hour).mean()
    data_std = df.groupby(lambda x : x.hour).std()
    
    ax[0,1].plot( data_mean.index,data_mean["fcr_price"], label=file)
    ax[0,1].fill_between(data_mean.index,data_mean["fcr_price"] - data_std["fcr_price"], data_mean["fcr_price"] +  data_std["fcr_price"], alpha=0.1)
    ax[0,1].set_xlabel("Time of day")
    ax[0,1].set_ylabel("€/kW/h")
    ax[0,1].set_title("FCR price profile ")




    df = pd.read_csv("data/spot_price/2020.csv")
    df["date"] = pd.date_range(start="1/1/2020", periods=len(df), freq="H")
    df.set_index("date", inplace=True, drop=True)
    df.plot(ax=ax[1,0])

    data_mean = df.groupby(lambda x : x.hour).mean()
    data_std = df.groupby(lambda x : x.hour).std()
    
    ax[1,1].plot( data_mean.index,data_mean["price_euros_mwh"], label=file)
    ax[1,1].fill_between(data_mean.index,data_mean["price_euros_mwh"] - data_std["price_euros_mwh"], data_mean["price_euros_mwh"] +  data_std["price_euros_mwh"], alpha=0.1)
 
    plt.show()
if __name__ == "__main__":
    plot_input_data()