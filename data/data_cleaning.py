import pandas as pd
import numpy as np

if __name__ == '__main__':
    # spx_close_1 = pd.read_excel('spx_memb_period_data.xlsx')
    # spx_close_2 = pd.read_excel('spx_memb_period_data2.xlsx')
    # spx_close_2 = pd.DataFrame(np.append(spx_close_2.columns.values, spx_close_2.values).reshape(1156, 4997))
    # spx_close_1.columns = range(5749)
    # spx_close_2.iloc[0, :] = spx_close_2.iloc[0, :].replace(r'Unnamed:+', np.nan, regex=True)
    # spx_close = pd.concat([spx_close_1, spx_close_2]).reset_index(drop=True)
    # raw_data = spx_close

    spx_volume = pd.read_excel('spx_memb_volume_data.xlsx')
    raw_data = spx_volume

    clean_data = []

    for i in range(int(2348/2)):
        stock_name = raw_data.iloc[2*i, 0].split()[0].replace('/', '-')
        close = raw_data.iloc[2*i:2*(i+1), 3:].T.dropna().set_index(2*i)
        close.index.name = 'Date'
        close.columns = [stock_name]
        clean_data.append(close)

    output_data = pd.concat(clean_data, axis=1).groupby(level=0, axis=1).sum().replace(0, np.nan)
    output_data.to_csv('spx_hist_volume.csv')

    # for i in range(output_data.shape[1]):
    #     stock_data = output_data.iloc[:, i].dropna()
    #     stock_data.index = pd.to_datetime(stock_data.index).date
    #     stock_name = stock_data.name
    #     stock_data.name = 'Close'
    #     stock_data.to_csv(f'stocks/{stock_name}.csv')
