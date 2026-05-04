from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

series=read_csv(r"C:\Users\Pc\Documents\TS_PROJECT\monthly-robberies.csv",header=0,index_col=0, parse_dates=True).squeeze()

#test ADF
def adf_test(serie):
    result=adfuller(serie)
    p_value=result[1]
    if p_value <0.05:
        print(f"p_value={p_value},la serie est stationnaire :on rejette H0")
    else:
        print(f"p_value={p_value},la serie n'est pas stationnaire :on accepte H0")

print("ADF:") 
adf_test(series)
#test kpss
def kpss_test():
    result= kpss(series, regression='c') 
    p_value=result[1]
    if p_value >0.05:
        print(f"p_value={p_value},la serie est stationnaire :on accepte H0")
    else:
        print(f"p_value={p_value},la serie n'est pas stationnaire :on refuse H0")

print("KPSS:")
kpss_test()

#power transformers
series_sqrt=np.sqrt(series)

series_log = np.log(series)
# plt.figure(1)
# plt.plot(series)
# plt.title("serie originale")
# plt.figure(2)
# plt.plot(series_log)
# plt.title("serie transformee")
# plt.show()

#differenciation

series_diff = series_log.diff()
series_diff = series.diff().dropna()
# plt.figure()
# plt.plot(series_diff)
# plt.title("serie diferenciee")
# plt.show()

#stationnarite apres transformation
print("----------------------------------------")
print("stationnarite pour la serie transformee:")
adf_test(series_diff)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Tracé de l'ACF
plot_acf(series_diff, ax=ax1, lags=40, title="Autocorrélation (ACF)")
# Tracé de la PACF
plot_pacf(series_diff, ax=ax2, lags=40, title="Autocorrélation Partielle (PACF)")
plt.show()

#decoupage de la serie
n = len(series_diff)
split_ratio = 0.8                        # 80% train, 20% test

split_index = int(n * split_ratio)

train = series_diff[:split_index]
test  =series_diff[split_index:]

# visualisation
plt.figure(figsize=(12, 4))
plt.plot(train.index, train, label="Train", color="steelblue")
plt.plot(test.index,  test,  label="Test",  color="orange")
plt.axvline(x=train.index[-1], color="red", linestyle="--", label="Coupure")
plt.title("Découpage Train / Test (chronologique)")
plt.legend()
plt.tight_layout()
plt.show()