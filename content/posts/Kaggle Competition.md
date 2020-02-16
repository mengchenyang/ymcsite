+++
title = "House Prices: Advanced Regression Techniques"
description = "Check out Mengchen's Machine Learning project that achieved a score within the top 12% in the Kaggle Competition."
tags = [
    "Machine Learning",
    "Feature Engineering",
    "Regression Analysis",
]
date = "2018-12-12"
categories = [
    "Data Science",
    "Machine Learning",
]
menu = "main"

+++

# **Mengchen's Kaggle Competition**

In this Kaggle Competition: 'House Prices: Advanced Regression Techniques'. This competition uses a Housing Dataset, which contains 1460 observations in both training and tests sets, and explanatory variables. The goal is to predict property Sale Price, hence this is a Regression problem.

My main focus was mainly ondata preparation, feature engineering and the building of a stacking model.

At the time of posting, this model achieved a score of 0.12, which is within the top 12% of the Leaderboard.

## **Check out the codes**

##### Loading and checking data

```python
# Load the libraries
import pandas as pd
from pandas import Series,DataFrame
from google.colab import files
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```



```python
# Load the data to dataframes
train_df = pd.read_csv("https://s3.amazonaws.com/it4ba/Kaggle/train.csv")
test_df  = pd.read_csv("https://s3.amazonaws.com/it4ba/Kaggle/test.csv")
```



```python
train_df.describe()
```



| Id    | MSSubClass  | LotFrontage | LotArea     | OverallQual   | OverallCond | YearBuilt   | YearRemodAdd | MasVnrArea  | BsmtFinSF1  | ...         | WoodDeckSF | OpenPorchSF | EnclosedPorch | 3SsnPorch   | ScreenPorch | PoolArea    | MiscVal     | MoSold       | YrSold      | SalePrice   |               |
| :---- | :---------- | :---------- | :---------- | :------------ | :---------- | :---------- | :----------- | :---------- | :---------- | :---------- | :--------- | :---------- | :------------ | :---------- | :---------- | :---------- | :---------- | :----------- | :---------- | :---------- | ------------- |
| count | 1460.000000 | 1460.000000 | 1201.000000 | 1460.000000   | 1460.000000 | 1460.000000 | 1460.000000  | 1460.000000 | 1452.000000 | 1460.000000 | ...        | 1460.000000 | 1460.000000   | 1460.000000 | 1460.000000 | 1460.000000 | 1460.000000 | 1460.000000  | 1460.000000 | 1460.000000 | 1460.000000   |
| mean  | 730.500000  | 56.897260   | 70.049958   | 10516.828082  | 6.099315    | 5.575342    | 1971.267808  | 1984.865753 | 103.685262  | 443.639726  | ...        | 94.244521   | 46.660274     | 21.954110   | 3.409589    | 15.060959   | 2.758904    | 43.489041    | 6.321918    | 2007.815753 | 180921.195890 |
| std   | 421.610009  | 42.300571   | 24.284752   | 9981.264932   | 1.382997    | 1.112799    | 30.202904    | 20.645407   | 181.066207  | 456.098091  | ...        | 125.338794  | 66.256028     | 61.119149   | 29.317331   | 55.757415   | 40.177307   | 496.123024   | 2.703626    | 1.328095    | 79442.502883  |
| min   | 1.000000    | 20.000000   | 21.000000   | 1300.000000   | 1.000000    | 1.000000    | 1872.000000  | 1950.000000 | 0.000000    | 0.000000    | ...        | 0.000000    | 0.000000      | 0.000000    | 0.000000    | 0.000000    | 0.000000    | 0.000000     | 1.000000    | 2006.000000 | 34900.000000  |
| 25%   | 365.750000  | 20.000000   | 59.000000   | 7553.500000   | 5.000000    | 5.000000    | 1954.000000  | 1967.000000 | 0.000000    | 0.000000    | ...        | 0.000000    | 0.000000      | 0.000000    | 0.000000    | 0.000000    | 0.000000    | 0.000000     | 5.000000    | 2007.000000 | 129975.000000 |
| 50%   | 730.500000  | 50.000000   | 69.000000   | 9478.500000   | 6.000000    | 5.000000    | 1973.000000  | 1994.000000 | 0.000000    | 383.500000  | ...        | 0.000000    | 25.000000     | 0.000000    | 0.000000    | 0.000000    | 0.000000    | 0.000000     | 6.000000    | 2008.000000 | 163000.000000 |
| 75%   | 1095.250000 | 70.000000   | 80.000000   | 11601.500000  | 7.000000    | 6.000000    | 2000.000000  | 2004.000000 | 166.000000  | 712.250000  | ...        | 168.000000  | 68.000000     | 0.000000    | 0.000000    | 0.000000    | 0.000000    | 0.000000     | 8.000000    | 2009.000000 | 214000.000000 |
| max   | 1460.000000 | 190.000000  | 313.000000  | 215245.000000 | 10.000000   | 9.000000    | 2010.000000  | 2010.000000 | 1600.000000 | 5644.000000 | ...        | 857.000000  | 547.000000    | 552.000000  | 508.000000  | 480.000000  | 738.000000  | 15500.000000 | 12.000000   | 2010.000000 | 755000.000000 |



```python
train_df.info()
```

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 81 columns):
Id               1460 non-null int64
MSSubClass       1460 non-null int64
MSZoning         1460 non-null object
LotFrontage      1201 non-null float64
LotArea          1460 non-null int64
Street           1460 non-null object
Alley            91 non-null object
LotShape         1460 non-null object
LandContour      1460 non-null object
Utilities        1460 non-null object
LotConfig        1460 non-null object
LandSlope        1460 non-null object
Neighborhood     1460 non-null object
Condition1       1460 non-null object
Condition2       1460 non-null object
BldgType         1460 non-null object
HouseStyle       1460 non-null object
OverallQual      1460 non-null int64
OverallCond      1460 non-null int64
YearBuilt        1460 non-null int64
YearRemodAdd     1460 non-null int64
RoofStyle        1460 non-null object
RoofMatl         1460 non-null object
Exterior1st      1460 non-null object
Exterior2nd      1460 non-null object
MasVnrType       1452 non-null object
MasVnrArea       1452 non-null float64
ExterQual        1460 non-null object
ExterCond        1460 non-null object
Foundation       1460 non-null object
BsmtQual         1423 non-null object
BsmtCond         1423 non-null object
BsmtExposure     1422 non-null object
BsmtFinType1     1423 non-null object
BsmtFinSF1       1460 non-null int64
BsmtFinType2     1422 non-null object
BsmtFinSF2       1460 non-null int64
BsmtUnfSF        1460 non-null int64
TotalBsmtSF      1460 non-null int64
Heating          1460 non-null object
HeatingQC        1460 non-null object
CentralAir       1460 non-null object
Electrical       1459 non-null object
1stFlrSF         1460 non-null int64
2ndFlrSF         1460 non-null int64
LowQualFinSF     1460 non-null int64
GrLivArea        1460 non-null int64
BsmtFullBath     1460 non-null int64
BsmtHalfBath     1460 non-null int64
FullBath         1460 non-null int64
HalfBath         1460 non-null int64
BedroomAbvGr     1460 non-null int64
KitchenAbvGr     1460 non-null int64
KitchenQual      1460 non-null object
TotRmsAbvGrd     1460 non-null int64
Functional       1460 non-null object
Fireplaces       1460 non-null int64
FireplaceQu      770 non-null object
GarageType       1379 non-null object
GarageYrBlt      1379 non-null float64
GarageFinish     1379 non-null object
GarageCars       1460 non-null int64
GarageArea       1460 non-null int64
GarageQual       1379 non-null object
GarageCond       1379 non-null object
PavedDrive       1460 non-null object
WoodDeckSF       1460 non-null int64
OpenPorchSF      1460 non-null int64
EnclosedPorch    1460 non-null int64
3SsnPorch        1460 non-null int64
ScreenPorch      1460 non-null int64
PoolArea         1460 non-null int64
PoolQC           7 non-null object
Fence            281 non-null object
MiscFeature      54 non-null object
MiscVal          1460 non-null int64
MoSold           1460 non-null int64
YrSold           1460 non-null int64
SaleType         1460 non-null object
SaleCondition    1460 non-null object
SalePrice        1460 non-null int64
dtypes: float64(3), int64(35), object(43)
memory usage: 924.0+ KB
```



```python
train_df.head
```

```
<bound method NDFrame.head of         Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
0        1          60       RL         65.0     8450   Pave   NaN      Reg   
1        2          20       RL         80.0     9600   Pave   NaN      Reg   
2        3          60       RL         68.0    11250   Pave   NaN      IR1   
3        4          70       RL         60.0     9550   Pave   NaN      IR1   
4        5          60       RL         84.0    14260   Pave   NaN      IR1   
5        6          50       RL         85.0    14115   Pave   NaN      IR1   
6        7          20       RL         75.0    10084   Pave   NaN      Reg   
7        8          60       RL          NaN    10382   Pave   NaN      IR1   
8        9          50       RM         51.0     6120   Pave   NaN      Reg   
9       10         190       RL         50.0     7420   Pave   NaN      Reg   
10      11          20       RL         70.0    11200   Pave   NaN      Reg   
11      12          60       RL         85.0    11924   Pave   NaN      IR1   
12      13          20       RL          NaN    12968   Pave   NaN      IR2   
13      14          20       RL         91.0    10652   Pave   NaN      IR1   
14      15          20       RL          NaN    10920   Pave   NaN      IR1   
15      16          45       RM         51.0     6120   Pave   NaN      Reg   
16      17          20       RL          NaN    11241   Pave   NaN      IR1   
17      18          90       RL         72.0    10791   Pave   NaN      Reg   
18      19          20       RL         66.0    13695   Pave   NaN      Reg   
19      20          20       RL         70.0     7560   Pave   NaN      Reg   
20      21          60       RL        101.0    14215   Pave   NaN      IR1   
21      22          45       RM         57.0     7449   Pave  Grvl      Reg   
22      23          20       RL         75.0     9742   Pave   NaN      Reg   
23      24         120       RM         44.0     4224   Pave   NaN      Reg   
24      25          20       RL          NaN     8246   Pave   NaN      IR1   
25      26          20       RL        110.0    14230   Pave   NaN      Reg   
26      27          20       RL         60.0     7200   Pave   NaN      Reg   
27      28          20       RL         98.0    11478   Pave   NaN      Reg   
28      29          20       RL         47.0    16321   Pave   NaN      IR1   
29      30          30       RM         60.0     6324   Pave   NaN      IR1   
...    ...         ...      ...          ...      ...    ...   ...      ...   
1430  1431          60       RL         60.0    21930   Pave   NaN      IR3   
1431  1432         120       RL          NaN     4928   Pave   NaN      IR1   
1432  1433          30       RL         60.0    10800   Pave  Grvl      Reg   
1433  1434          60       RL         93.0    10261   Pave   NaN      IR1   
1434  1435          20       RL         80.0    17400   Pave   NaN      Reg   
1435  1436          20       RL         80.0     8400   Pave   NaN      Reg   
1436  1437          20       RL         60.0     9000   Pave   NaN      Reg   
1437  1438          20       RL         96.0    12444   Pave   NaN      Reg   
1438  1439          20       RM         90.0     7407   Pave   NaN      Reg   
1439  1440          60       RL         80.0    11584   Pave   NaN      Reg   
1440  1441          70       RL         79.0    11526   Pave   NaN      IR1   
1441  1442         120       RM          NaN     4426   Pave   NaN      Reg   
1442  1443          60       FV         85.0    11003   Pave   NaN      Reg   
1443  1444          30       RL          NaN     8854   Pave   NaN      Reg   
1444  1445          20       RL         63.0     8500   Pave   NaN      Reg   
1445  1446          85       RL         70.0     8400   Pave   NaN      Reg   
1446  1447          20       RL          NaN    26142   Pave   NaN      IR1   
1447  1448          60       RL         80.0    10000   Pave   NaN      Reg   
1448  1449          50       RL         70.0    11767   Pave   NaN      Reg   
1449  1450         180       RM         21.0     1533   Pave   NaN      Reg   
1450  1451          90       RL         60.0     9000   Pave   NaN      Reg   
1451  1452          20       RL         78.0     9262   Pave   NaN      Reg   
1452  1453         180       RM         35.0     3675   Pave   NaN      Reg   
1453  1454          20       RL         90.0    17217   Pave   NaN      Reg   
1454  1455          20       FV         62.0     7500   Pave  Pave      Reg   
1455  1456          60       RL         62.0     7917   Pave   NaN      Reg   
1456  1457          20       RL         85.0    13175   Pave   NaN      Reg   
1457  1458          70       RL         66.0     9042   Pave   NaN      Reg   
1458  1459          20       RL         68.0     9717   Pave   NaN      Reg   
1459  1460          20       RL         75.0     9937   Pave   NaN      Reg   

​     LandContour Utilities    ...     PoolArea PoolQC  Fence MiscFeature  \
0            Lvl    AllPub    ...            0    NaN    NaN         NaN   
1            Lvl    AllPub    ...            0    NaN    NaN         NaN   
2            Lvl    AllPub    ...            0    NaN    NaN         NaN   
3            Lvl    AllPub    ...            0    NaN    NaN         NaN   
4            Lvl    AllPub    ...            0    NaN    NaN         NaN   
5            Lvl    AllPub    ...            0    NaN  MnPrv        Shed   
6            Lvl    AllPub    ...            0    NaN    NaN         NaN   
7            Lvl    AllPub    ...            0    NaN    NaN        Shed   
8            Lvl    AllPub    ...            0    NaN    NaN         NaN   
9            Lvl    AllPub    ...            0    NaN    NaN         NaN   
10           Lvl    AllPub    ...            0    NaN    NaN         NaN   
11           Lvl    AllPub    ...            0    NaN    NaN         NaN   
12           Lvl    AllPub    ...            0    NaN    NaN         NaN   
13           Lvl    AllPub    ...            0    NaN    NaN         NaN   
14           Lvl    AllPub    ...            0    NaN   GdWo         NaN   
15           Lvl    AllPub    ...            0    NaN  GdPrv         NaN   
16           Lvl    AllPub    ...            0    NaN    NaN        Shed   
17           Lvl    AllPub    ...            0    NaN    NaN        Shed   
18           Lvl    AllPub    ...            0    NaN    NaN         NaN   
19           Lvl    AllPub    ...            0    NaN  MnPrv         NaN   
20           Lvl    AllPub    ...            0    NaN    NaN         NaN   
21           Bnk    AllPub    ...            0    NaN  GdPrv         NaN   
22           Lvl    AllPub    ...            0    NaN    NaN         NaN   
23           Lvl    AllPub    ...            0    NaN    NaN         NaN   
24           Lvl    AllPub    ...            0    NaN  MnPrv         NaN   
25           Lvl    AllPub    ...            0    NaN    NaN         NaN   
26           Lvl    AllPub    ...            0    NaN    NaN         NaN   
27           Lvl    AllPub    ...            0    NaN    NaN         NaN   
28           Lvl    AllPub    ...            0    NaN    NaN         NaN   
29           Lvl    AllPub    ...            0    NaN    NaN         NaN   
...          ...       ...    ...          ...    ...    ...         ...   
1430         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1431         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1432         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1433         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1434         Low    AllPub    ...            0    NaN    NaN         NaN   
1435         Lvl    AllPub    ...            0    NaN  GdPrv         NaN   
1436         Lvl    AllPub    ...            0    NaN   GdWo         NaN   
1437         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1438         Lvl    AllPub    ...            0    NaN  MnPrv         NaN   
1439         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1440         Bnk    AllPub    ...            0    NaN    NaN         NaN   
1441         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1442         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1443         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1444         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1445         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1446         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1447         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1448         Lvl    AllPub    ...            0    NaN   GdWo         NaN   
1449         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1450         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1451         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1452         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1453         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1454         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1455         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1456         Lvl    AllPub    ...            0    NaN  MnPrv         NaN   
1457         Lvl    AllPub    ...            0    NaN  GdPrv        Shed   
1458         Lvl    AllPub    ...            0    NaN    NaN         NaN   
1459         Lvl    AllPub    ...            0    NaN    NaN         NaN   

​     MiscVal MoSold YrSold  SaleType  SaleCondition  SalePrice  
0          0      2   2008        WD         Normal     208500  
1          0      5   2007        WD         Normal     181500  
2          0      9   2008        WD         Normal     223500  
3          0      2   2006        WD        Abnorml     140000  
4          0     12   2008        WD         Normal     250000  
5        700     10   2009        WD         Normal     143000  
6          0      8   2007        WD         Normal     307000  
7        350     11   2009        WD         Normal     200000  
8          0      4   2008        WD        Abnorml     129900  
9          0      1   2008        WD         Normal     118000  
10         0      2   2008        WD         Normal     129500  
11         0      7   2006       New        Partial     345000  
12         0      9   2008        WD         Normal     144000  
13         0      8   2007       New        Partial     279500  
14         0      5   2008        WD         Normal     157000  
15         0      7   2007        WD         Normal     132000  
16       700      3   2010        WD         Normal     149000  
17       500     10   2006        WD         Normal      90000  
18         0      6   2008        WD         Normal     159000  
19         0      5   2009       COD        Abnorml     139000  
20         0     11   2006       New        Partial     325300  
21         0      6   2007        WD         Normal     139400  
22         0      9   2008        WD         Normal     230000  
23         0      6   2007        WD         Normal     129900  
24         0      5   2010        WD         Normal     154000  
25         0      7   2009        WD         Normal     256300  
26         0      5   2010        WD         Normal     134800  
27         0      5   2010        WD         Normal     306000  
28         0     12   2006        WD         Normal     207500  
29         0      5   2008        WD         Normal      68500  
...      ...    ...    ...       ...            ...        ...  
1430       0      7   2006        WD         Normal     192140  
1431       0     10   2009        WD         Normal     143750  
1432       0      8   2007        WD         Normal      64500  
1433       0      5   2008        WD         Normal     186500  
1434       0      5   2006        WD         Normal     160000  
1435       0      7   2008       COD        Abnorml     174000  
1436       0      5   2007        WD         Normal     120500  
1437       0     11   2008       New        Partial     394617  
1438       0      4   2010        WD         Normal     149700  
1439       0     11   2007        WD         Normal     197000  
1440       0      9   2008        WD         Normal     191000  
1441       0      5   2008        WD         Normal     149300  
1442       0      4   2009        WD         Normal     310000  
1443       0      5   2009        WD         Normal     121000  
1444       0     11   2007        WD         Normal     179600  
1445       0      5   2007        WD         Normal     129000  
1446       0      4   2010        WD         Normal     157900  
1447       0     12   2007        WD         Normal     240000  
1448       0      5   2007        WD         Normal     112000  
1449       0      8   2006        WD        Abnorml      92000  
1450       0      9   2009        WD         Normal     136000  
1451       0      5   2009       New        Partial     287090  
1452       0      5   2006        WD         Normal     145000  
1453       0      7   2006        WD        Abnorml      84500  
1454       0     10   2009        WD         Normal     185000  
1455       0      8   2007        WD         Normal     175000  
1456       0      2   2010        WD         Normal     210000  
1457    2500      5   2010        WD         Normal     266500  
1458       0      4   2010        WD         Normal     142125  
1459       0      6   2008        WD         Normal     147500  

[1460 rows x 81 columns]>


```



```python
sns.distplot(train_df["SalePrice"])
```

```
<matplotlib.axes._subplots.AxesSubplot at 0x7f7da23e9e10>
```

![img](data:image/png;base64,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%0AEEJ4if8HgXngyFOHDPQAAAAASUVORK5CYII=%0A)

```python
train_df["SalePrice"].describe()
```

```
count      1460.000000
mean     180921.195890
std       79442.502883
min       34900.000000
25%      129975.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
Name: SalePrice, dtype: float64
```

```python
train_df.shape, test_df.shape
```

```python
((1460, 81), (1459, 80))
```

##### Features engineering

```python
# Load the data to dataframes for features engineering
train = pd.read_csv("https://s3.amazonaws.com/it4ba/Kaggle/train.csv", index_col = 0)
test = pd.read_csv("https://s3.amazonaws.com/it4ba/Kaggle/test.csv", index_col = 0)
```

```python
# Concatenate both train and test values, and drop SalePrice
all_df = pd.concat((train, test),axis = 0)
all_df.drop(["SalePrice"], axis = 1, inplace = True)
```

```python
all_df.shape
```

```
(2919, 79)
```

```python
# First find all the missing data
all_df.isnull().sum().sort_values(ascending = False).head(40)
```

```
PoolQC          2909
MiscFeature     2814
Alley           2721
Fence           2348
FireplaceQu     1420
LotFrontage      486
GarageFinish     159
GarageQual       159
GarageYrBlt      159
GarageCond       159
GarageType       157
BsmtCond          82
BsmtExposure      82
BsmtQual          81
BsmtFinType2      80
BsmtFinType1      79
MasVnrType        24
MasVnrArea        23
MSZoning           4
BsmtHalfBath       2
Utilities          2
Functional         2
BsmtFullBath       2
Electrical         1
Exterior2nd        1
KitchenQual        1
GarageCars         1
Exterior1st        1
GarageArea         1
TotalBsmtSF        1
BsmtUnfSF          1
BsmtFinSF2         1
BsmtFinSF1         1
SaleType           1
Condition2         0
FullBath           0
2ndFlrSF           0
3SsnPorch          0
BedroomAbvGr       0
BldgType           0
dtype: int64
```

There are missing data in both numerical and categorical data and all of those data could fall in two types

- The null means the house doesnt have such features, such as "PoolQC", "MiscFeature" or "Alley", for those kind of data, I choose to fillin none or zero.
- The null means missing values, for those kind of data, I choose to fillin the most common value.

```python
# For the following categorical features, null values mean the house doesn't have such features, so I fillin none
# for example, "PoolQC" with null means the house doesn't have a pool
null_none = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "GarageType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType"]
```

```python
for i in null_none:
   all_df[i].fillna("None", inplace=True)
```

```python
# For the following numerical features, null values mean the house doesn't have such features, so I fillin 0
# for example, GarageYrBlt with null means the house doesn't have a garage
null_zero = ["GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"]
```

```python
for i in null_zero:
   all_df[i].fillna(0, inplace=True)
```

```python
all_df.isnull().sum().sort_values(ascending = False).head(10)
```

```
LotFrontage    486
MSZoning         4
Utilities        2
Functional       2
Exterior1st      1
Exterior2nd      1
SaleType         1
KitchenQual      1
Electrical       1
YrSold           0
dtype: int64
```

```python
# Because the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , I choose to fillin missing values with the average LotFrontage within the same neighborhood
all_df["LotFrontage"] = all_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.mean()))
```

For the following features, the number of missing values is very small, so I choose to fillin null with the most common values.

```python
all_df["MSZoning"] = all_df["MSZoning"].fillna(all_df["MSZoning"].mode()[0])
```

```python
all_df["Utilities"] = all_df["Utilities"].fillna(all_df["Utilities"].mode()[0])
```

```python
all_df["Functional"] = all_df["Functional"].fillna(all_df["Functional"].mode()[0])
```

```python
all_df["Exterior1st"] = all_df["Exterior1st"].fillna(all_df["Exterior1st"].mode()[0])
```

```python
all_df["Exterior2nd"] = all_df["Exterior2nd"].fillna(all_df["Exterior2nd"].mode()[0])
```

```python
all_df["SaleType"] = all_df["SaleType"].fillna(all_df["SaleType"].mode()[0])
```

```python
all_df["KitchenQual"] = all_df["KitchenQual"].fillna(all_df["KitchenQual"].mode()[0])
```

```python
all_df["Electrical"] = all_df["Electrical"].fillna(all_df["Electrical"].mode()[0])
```

```python
# Check the number of the null values
all_df.isnull().sum().sum()
```

```
0
```

There are few numerical features that infact categorical. Such as MSSubClass means The building class. Coverting those data into categorical features. So that those data won't mislead the regression model.

```python
# Change the datatype to String
all_df["MSSubClass"] = all_df["MSSubClass"].astype(str)
```

```python
all_df["OverallCond"] = all_df["OverallCond"].astype(str)
```

```python
all_df["YrSold"] = all_df["YrSold"].astype(str)
```

```python
all_df["MoSold"] = all_df["MoSold"].astype(str)
```

```python
# Check the data type
all_df["MSSubClass"].dtypes, all_df["OverallCond"].dtypes, all_df["YrSold"].dtypes, all_df["MoSold"].dtypes,
```

```
(dtype('O'), dtype('O'), dtype('O'), dtype('O'))
```

##### Data Normalization

The log transformation is the most popular methods of transformations used to transform skewed data to approximately conform to normality. As the distribution ploted ealier, the data set is highly skewed and a log transformation would make the data more normally distributted and easier for the regression models to make predictions.

```python
# Load the library
from scipy.stats import skew
```

```python
# Find all numeric features
quantity = all_df.dtypes[all_df.dtypes != "object"].index
```

```python
# Compute skewness
skew_df = all_df[quantity].apply(lambda x: skew(x.dropna()))
# Identify Highly skewed numeric features
skew_df = skew_df[skew_df > 0.75]
# Index the data set
skew_df = skew_df.index
```

```python
#Log transform all skewed numeric features
all_df[skew_df] = np.log1p(all_df[skew_df])
```

Getting dummy categorical features of the data set

```python
all_dummy = pd.get_dummies(all_df)
```

```python
# Check the data 
all_dummy.shape
```

```
(2919, 340)
```

```python
# Becasue we concatenate the train and test data, now I will split it up
dummy_train = all_dummy[:train.shape[0]]
dummy_test = all_dummy[train.shape[0]:]
```

```python
# Check the data after creating
dummy_train.head()
```

| 1stFlrSF | 2ndFlrSF | 3SsnPorch | BedroomAbvGr | BsmtFinSF1 | BsmtFinSF2 | BsmtFullBath | BsmtHalfBath | BsmtUnfSF | EnclosedPorch | ...      | SaleType_WD | Street_Grvl | Street_Pave | Utilities_AllPub | Utilities_NoSeWa | YrSold_2006 | YrSold_2007 | YrSold_2008 | YrSold_2009 | YrSold_2010 |      |
| :------- | :------- | :-------- | :----------- | :--------- | :--------- | :----------- | :----------- | :-------- | :------------ | :------- | :---------- | :---------- | :---------- | :--------------- | :--------------- | :---------- | :---------- | :---------- | :---------- | :---------- | :--- |
| Id       |          |           |              |            |            |              |              |           |               |          |             |             |             |                  |                  |             |             |             |             |             |      |
| 1        | 6.753438 | 6.751101  | 0.0          | 3          | 6.561031   | 0.0          | 1.0          | 0.000000  | 5.017280      | 0.000000 | ...         | 1           | 0           | 1                | 1                | 0           | 0           | 0           | 1           | 0           | 0    |
| 2        | 7.141245 | 0.000000  | 0.0          | 3          | 6.886532   | 0.0          | 0.0          | 0.526589  | 5.652489      | 0.000000 | ...         | 1           | 0           | 1                | 1                | 0           | 0           | 1           | 0           | 0           | 0    |
| 3        | 6.825460 | 6.765039  | 0.0          | 3          | 6.188264   | 0.0          | 1.0          | 0.000000  | 6.075346      | 0.000000 | ...         | 1           | 0           | 1                | 1                | 0           | 0           | 0           | 1           | 0           | 0    |
| 4        | 6.869014 | 6.629363  | 0.0          | 3          | 5.379897   | 0.0          | 1.0          | 0.000000  | 6.293419      | 1.888504 | ...         | 1           | 0           | 1                | 1                | 0           | 1           | 0           | 0           | 0           | 0    |
| 5        | 7.044033 | 6.960348  | 0.0          | 4          | 6.486161   | 0.0          | 1.0          | 0.000000  | 6.196444      | 0.000000 | ...         | 1           | 0           | 1                | 1                | 0           | 0           | 0           | 1           | 0           | 0    |

5 rows × 340 columns

##### Building Stack Models

```python
# Load the libraries
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score
```

As I mentioned before, the data need to be normalized befor the model run. Because I dropped the SalePrice from the dataset used earlier, now I log transform the SalePrice.

```python
# Check the skewness
sns.distplot(train["SalePrice"])
```

<matplotlib.axes._subplots.AxesSubplot at 0x7f7d9ae0c5c0>

![img](data:image/png;base64,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%0AEEJ4if8HgXngyFOHDPQAAAAASUVORK5CYII=%0A)

```python
# Log transform 
sns.distplot(np.log1p(train["SalePrice"]))
```

<matplotlib.axes._subplots.AxesSubplot at 0x7f7d9a6e2828>

![img](data:image/png;base64,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%0A)

The data is more normally distributted after the transformation

Creating train and test data

```python
X_train = dummy_train.values
X_test = dummy_test.values
y_train = np.log1p(train["SalePrice"])
```

```python
# Define Root Mean Square Error and use rmse and 5-folds cross-validation to evaluate predictions
def rmse_cv (model, X_train, y_train):
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return rmse
```

5 models from the project requirements
Decision Tree
Random Forests
Regularized linear regression - ridge
Regularized linear regression - lasso
XGboost

```python
# Define models
dt = DecisionTreeRegressor(random_state= 1)
rf = RandomForestRegressor(random_state= 1)
ridge = Ridge(normalize = True)
lasso = Lasso(normalize=True)
xgb = XGBRegressor()
```

Decision Tree

The decision tree splits on values of features that generates a group the all data in that group share similar features. It will keep doing this until all the leaf nodes are perfectly pure, or certain stopping mechanisms are met. The parameters are basically those stopping parameters.In terms of regression, what the tree will do is take the average of all true y of each leaf as the estimated y for that particular path, so that when it predicts the test dataset, each record from that test dataset will basically follow some path down the tree until it hits a leaf node, and the estimated y for that record will be the average true y of all observations in that leaf node.

```python
# Hyperparameter tuning
# Define the grid of hyperparameters params_dt
params_dt = {"max_depth": [3, 4, 5, 6], "min_samples_leaf": [0.04, 0.06, 0.08], "max_features": [0.2, 0.4, 0.6, 0.8]}

# Start a 10-fold CV grid search  
grid_dt = GridSearchCV(estimator = dt, param_grid = params_dt, scoring = "neg_mean_squared_error", cv = 10, n_jobs = -1)
```

```python
# Fit grid_dt to the training data
grid_dt.fit(X_train, y_train)

# predict with the best estimators
y_dt = grid_dt.best_estimator_.predict(X_train)
```

```python
# Evaluate the set rmse
dt_rmse_ht = mse(y_train, y_dt)**1/2
dt_rmse_ht
```

```
0.018332157877120645
```

```python
# Prediction without hyperparameter tuning
dt_rmse = rmse_cv(dt, X_train, y_train).mean()
dt_rmse
```

```
0.20679377020259698
```

Random Forests

The random forest is basically a collection of decision trees which use a subset of the training data to do the training. These trees are usually not as deep as a single decision tree model, which helps alleviate the overfitting symptoms of a single decision tree. The idea of random forest is that the using of many weak learners can generalize the data well and therefore less overfit.

```python
# Hyperparameter tuning
# Define the grid of hyperparameter params_rf
params_rf = {"n_estimators": [300, 400, 500], "max_depth": [3, 4, 5, 6], "min_samples_leaf": [0.04, 0.06, 0.08], "max_features": [0.2, 0.4, 0.6, 0.8]}

# Start a 3-fold CV grid search 
grid_rf = GridSearchCV(estimator = rf, param_grid = params_rf, cv = 3, scoring = "neg_mean_squared_error", verbose = 1, n_jobs = -1)
```

```python
# Fit grid_rf to the training data
grid_rf.fit(X_train, y_train)

# predict with the best estimators
y_rf = grid_rf.best_estimator_.predict(X_train)
```

```python
# Evaluate the set rmse
rf_rmse_ht = mse(y_train, y_rf)**1/2
rf_rmse_ht
```

```
0.014709754114310037
```

```python
# Prediction without hyperparameter tuning
rf_rmse = rmse_cv(rf, X_train, y_train).mean()
rf_rmse
```

```
0.15727050726277095
```

Regularized linear regression - ridge

The ridge regression analyzing multiple regression data that suffer from multicollinearity. When multicollinearity occurs, least squares estimates are unbiased, but their variances are large so they may be far from the true value. By adding a degree of bias to the regression estimates, ridge regression reduces the standard errors.

```python
# Hyperparameter tuning
# Define the grid of hyperparameter params_ridge
params_ridge = {"alpha": [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.3, 1, 3, 10, 15, 30, 100]}

# Start a 10-fold CV grid search 
grid_ridge = GridSearchCV(estimator = ridge, param_grid = params_ridge, cv = 10, scoring="neg_mean_squared_error", verbose=1, n_jobs=-1)
```

```python
# Fit grid_ridge to the training data
grid_ridge.fit(X_train, y_train)

# predict with the best estimators
y_ridge = grid_ridge.best_estimator_.predict(X_train)
```

Fitting 10 folds for each of 14 candidates, totalling 140 fits.

```python
# Evaluate the set rmse
ridge_rmse_ht = mse(y_train, y_ridge)**1/2
ridge_rmse_ht
```

```
0.005036632843243046
```

```python
# Prediction without hyperparameter tuning
ridge_rmse = rmse_cv(ridge, X_train, y_train).mean()
ridge_rmse
```

```
0.13631503230862035
```

Regularized linear regression - lasso

The LASSO (Least Absolute Shrinkage and Selection Operator) is a regression method that involves penalizing the absolute size of the regression coefficients. By penalizing you end up in a situation where some of the parameter estimates may be exactly zero. The larger the penalty applied, the further estimates are shrunk towards zero. This is convenient when we want some automatic feature/variable selection, or when dealing with highly correlated predictors.

```python
# Hyperparameter tuning
# Define the grid of hyperparameter params_lasso
params_lasso = {"alpha": [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.3, 1, 3, 10, 15, 30, 100],"max_iter":[10000]}

# Start a 10-fold CV grid search 
grid_lasso = GridSearchCV(estimator = lasso, param_grid = params_lasso, cv = 10, scoring="neg_mean_squared_error", verbose=1, n_jobs=-1)
```

```python
# Fit grid_lasso to the training data
grid_lasso.fit(X_train, y_train)

# predict with the best estimators
y_lasso = grid_lasso.best_estimator_.predict(X_train)
```

Fitting 10 folds for each of 14 candidates, totalling 140 fits

```python
# Evaluate the set rmse
lasso_rmse_ht = mse(y_train, y_lasso)**1/2
lasso_rmse_ht
```

```
0.005143980563639151
```

```python
# Prediction without hyperparameter tuning
lasso_rmse = rmse_cv(lasso, X_train, y_train).mean()
lasso_rmse
```

```
0.39922568603836983
```


XGboost

XGBoost is an algorithm that has recently been dominating applied machine learning for structured or tabular data. XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.

```python
# Hyperparameter tuning
# Define the grid of hyperparameter params_lasso
params_xgb = {"n_estimators":[300, 400, 500], "learning_rate":[0.0001, 0.001, 0.01, 0.1], "max_depth":[3, 4, 5, 6], "min_child_weight":[1, 1.5, 2]}

# Start a 10-fold CV grid search 
grid_xgb = GridSearchCV(estimator = xgb, param_grid = params_xgb, cv = 3, scoring="neg_mean_squared_error", verbose=1, n_jobs=-1)
```

```python
# Fit grid_lasso to the training data
grid_xgb.fit(X_train, y_train)

# predict with the best estimators
y_xgb = grid_xgb.best_estimator_.predict(X_train)
```

Fitting 3 folds for each of 144 candidates, totalling 432 fits.

```python
# Evaluate the set rmse
xgb_rmse_ht = mse(y_train, y_xgb)**1/2
xgb_rmse_ht
```

```
0.0018037046767226386
```

```python
# Prediction without hyperparameter tuning
xgb_rmse = rmse_cv(xgb, X_train, y_train).mean()
xgb_rmse
```

```
0.1318782590776898
```

```python
# Compare the results of different models
d = {"Decision Tree": [dt_rmse, dt_rmse_ht], "Random Forests": [rf_rmse, rf_rmse_ht], "Ridge": [ridge_rmse, ridge_rmse_ht], "Lasso": [lasso_rmse, lasso_rmse_ht],"XGboost": [xgb_rmse, xgb_rmse_ht]}
title = ['rmse', 'rmse with hyperparameter tuning']
results = pd.DataFrame(data = d, index = title )
results
```

| Decision Tree                   | Lasso    | Random Forests | Ridge    | XGboost  |          |
| :------------------------------ | :------- | :------------- | :------- | :------- | -------- |
| rmse                            | 0.206794 | 0.399226       | 0.157271 | 0.136315 | 0.131878 |
| rmse with hyperparameter tuning | 0.018332 | 0.005144       | 0.014710 | 0.005037 | 0.001804 |

Lasso and XGBoost have the lowest RMSE so I choose those two model to make my final submission.

```python
# Because I made log transformation to the y_train data set, now it has to be changed back with np.expm1
Y_lasso = np.expm1(grid_lasso.best_estimator_.predict(X_test))
Y_xgb = np.expm1(grid_xgb.best_estimator_.predict(X_test))
# Take the average of two models' predictions
Y_pred = (Y_xgb + Y_lasso) / 2
```

```python
# prepare a submission file
submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": Y_pred
    })
submission.to_csv('dt1.csv', index=False)
files.download('dt1.csv')
```

##### Submission!

## **Conclusion**

I have performed primary data preparation techniques before applying several regression models, and then combining their performance into a stacked model. This achieved a final RMSE about 0.12 which put me within the top 12% of the leaderboard.

Some potential improvements:

- Applying it to a deep learning model like TensorFlow
- Performing a more rigorous GridSearchCV
- Exploring more complex stack models for better final prediction.

##### Thank you for reading!

