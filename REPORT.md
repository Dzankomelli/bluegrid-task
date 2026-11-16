# Forecast of A

## Management summary

**What I tried and why:** The analysis of the data showed four things a model has to handle: A has rare but extreme spikes, follows a strong daily and weekly pattern, depends on the drivers in a non-linear way, and its level drifted downwards over the years. I therefore compared three models: a linear regression per hour (simple, explainable, and able to extrapolate), gradient-boosted trees (able to learn non-linear steps), and a combination of the two. All are fitted on a transformed A that dampens the spikes, refitted every day on recent data only, tuned on one validation year, and tested once on the final two years, which played no part in building them.

**Results:** Gradient-boosted trees are the best model, with an average error of **3.54** units per hour. Simply repeating recent values gives 8.04, so the model removes **56 %** of that error. On the same data and test period, two published state-of-the-art reference models reach 3.61 and 3.41, so the model is on par with them, although it is a single model.

**Limitation:** The models were tuned on MAE and judged mainly on MAE and RMSE, because the goal was the best average forecast of every hour, which suits trading the normal daily profile. In the most extreme hours the error is about three times the average. If the value lies in forecasting strong outliers, the next step is a custom metric that measures exactly those errors, with the model tuned on it (section 6). Every number in this report comes from one run of the delivered code.

## 1. What is known at forecast time

The forecast for day T is made on day T-1 before 12:11. By then A is known up to 23:00 of day T-1, and N_1 and N_2 are known up to 23:00 of day T. The code keeps to this in two places:

- **Inputs:** Each hour of day T gets N_1 and N_2 of day T, but A only from day T-1 or earlier.
- **Training:** The model that forecasts day T is fitted only on days before T, and its settings were chosen before the test period starts.

For example, Wednesday's forecast uses N_1 and N_2 for Wednesday, A up to Tuesday 23:00, and a model fitted on data up to Tuesday 23:00. Two automated tests enforce this: they alter every value not yet known and check that neither the inputs nor the forecast of day T change.

## 2. The data and the benchmark

The file holds 52,416 hourly rows from 2012-01-09 to 2017-12-31, with no missing or duplicated hours. Data this clean and regular is unusual for an operational source, so I suspected a public benchmark and searched for it. It is, row for row, the German dataset of Lago et al. (2021), "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark". A is the day-ahead power price, N_1 a load forecast and N_2 the wind and solar generation forecast. The code checks the match on every run.

**Why compare with it:** Beating a naive forecast only shows that a model has learned something, not whether it is good. The study is the standard reference for this problem, and its authors publish the forecasts of their best models for the test period. That gives an independent answer to "how good is good". I used the study in two ways:

- **As a yardstick:** I adopted its test period and scored its two published forecasts, a LASSO-based linear ensemble (LEAR) and a neural-network ensemble (DNN), with my own code on exactly the same hours, so the comparison in section 6 is like for like. My models use only the provided file.
- **Not as a recipe:** I did not rebuild the study's models but chose mine for what the data needs. My linear model is similar to the study's; I kept it because it extrapolates beyond the values seen in training. But prices do not respond linearly at the edges: they jump when supply is nearly exhausted and collapse, even below zero, when there is too much of it. So I added tree models, which can follow such steps, and used a small, explainable set of features instead of the study's large set of lagged hourly inputs.

| Period | Dates | Use |
|---|---|---|
| Train | 2012-01-09 to 2015-01-04 | fitting, feature filter |
| Validation | 2015-01-05 to 2016-01-03 | choosing settings |
| Test | 2016-01-04 to 2017-12-31 | one final run |

## 3. Exploratory analysis

Only data before the test period was analysed (full analysis in results/eda/EDA.md). Each finding led to one design decision:

- **Heavy tails:** A ranges from -222 to 210 around a median of 35; 0.9 % of hours are zero or negative; kurtosis 19.9. → *Transform A before fitting.*
- **Strong daily and weekly pattern:** The hourly average moves from 23 at 03:00 to 49 at 19:00; weekdays average 38 to 40, Saturday 31, Sunday 24. → *Hour and weekday as inputs; A of yesterday and last week as features.*
- **N_1 - N_2 is the best single driver** (rank correlation 0.72), and the relation bends at both ends. → *Use the difference as a feature; allow non-linear models.*
- **The data drifts:** From 2012 to 2015 the yearly mean of N_1 - N_2 fell from 14,622 to 8,887, and that of A from 42.9 to 31.6. → *Train on recent data and refit often.*
- **Naive baseline:** "Yesterday for Tuesday to Friday, last week otherwise" has an MAE of 7.54, against 8.76 for yesterday and 8.83 for last week alone. → *Use it as the benchmark to beat; it is also the one used in the literature.*

## 4. Features

Nineteen features per hour, in five groups:

| Group | Features | Why |
|----------|-----------------|--------------|
| Drivers on day T | N_1, N_2, N_1 - N_2, N_2 / N_1, daily mean of N_1 - N_2 | Level A is set against |
| Change in drivers | N_1 - N_2 against yesterday and last week; N_1, N_2 against their 4-week same-hour average | How day T differs from days with known A |
| Recent A | same hour 1, 2 and 7 days ago; yesterday's mean, min and max | Latest level and shape of A |
| Was yesterday unusual | yesterday's A minus its 7-day and 4-week same-weekday average | Don't carry a one-off spike forward |
| Calendar | hour, weekday | Daily and weekly cycle |

Ratios are used only for N_1 and N_2, which are always positive; A crosses zero, so it enters as levels and differences. Of any two features with a rank correlation above 0.95 on the training period, the one less related to A is dropped. This removed only N_2, whose information remains in the difference and the ratio.

## 5. Models

**Preparing A:** Spikes and negative values would dominate any fit. So A is centred on its median, scaled by its median absolute deviation and passed through asinh, which dampens spikes like a logarithm but also works for negative values. This reduces the kurtosis from 19.9 to -0.5. Forecasts are converted back to original units.

**Three models,** each chosen for what it does well:

- **Linear:** one LASSO regression per hour, because the drivers act differently at night and at peak. Explainable, and keeps responding when the drivers leave the training range, which matters as the data drifts. Cannot follow steps.
- **Boosted trees:** one XGBoost model for all hours, with the hour as an input. Learns bends, steps and interactions without being told their shape, but its forecast stays flat beyond the training range.
- **Linear + boosted:** the linear model plus trees trained on its errors, meant to combine the reach of the one with the shape of the other.

**Tuning:** As in operation, each model is refitted for every forecast day on earlier days only. On the validation year I tried windows of one year, two years and all history, weekly and daily refits, LASSO penalties 0.001 to 0.03 and tree depths 3 and 6. The best settings were frozen: all models refit daily, the linear one on the last 364 days (validation MAE 3.51), the trees and the combination on the last 728 days (3.45 and 3.50). Training on all history was never the best choice, which confirms the drift. Refitting daily instead of weekly improved every model by about 1 %.

## 6. Results and comparison with the benchmark

The test period is scored with several measures. **MAE**, the main one, is the average size of a miss. **RMSE** squares the misses, so a few very bad hours raise it sharply. **rMAE** is MAE relative to the naive forecast; below 1 beats naive. **Bias** is the average of forecast minus actual. **Low / High 5 %** is the MAE in the hours where A was among its lowest and highest 5 %, which the overall average hides.

| Forecast | MAE | RMSE | rMAE | Bias | Low 5 % | High 5 % |
|---|---|---|---|---|---|---|
| Naive | 8.04 | 13.87 | 1.00 | -0.02 | 24.41 | 16.03 |
| Linear, N_1 and N_2 only | 5.53 | 8.70 | 0.69 | -0.89 | 11.24 | 19.02 |
| Linear | 4.66 | 13.50 | 0.58 | 0.66 | 13.03 | 24.37 |
| **Boosted trees** | **3.54** | **6.05** | **0.44** | **-0.35** | **9.52** | **10.59** |
| Linear + boosted | 3.96 | 8.23 | 0.49 | 0.16 | 10.55 | 14.90 |
| *Published: LEAR ensemble* | 3.61 | 6.51 | 0.45 | -0.08 | 11.94 | 10.67 |
| *Published: DNN ensemble* | **3.41** | **5.93** | **0.42** | **-0.27** | **8.83** | **9.82** |

- **Boosted trees are the model to use.** Best of my models on validation, and on test on MAE, RMSE and rMAE.
- **They are on par with the published models:** MAE 2 % below the linear ensemble and 4 % above the neural-network ensemble. Each of those averages four models while mine is one, so averaging the trees with a model that errs differently is the obvious next gain. Significance was not tested (e.g. Diebold-Mariano), so this means "on par", not a ranking.
- **The features matter.** "Linear, N_1 and N_2 only" is a reference run, not a fourth model: the linear model given only N_1, N_2, hour and weekday. Its MAE is 5.53, against 4.66 with the features of section 4.
- **The linear model failed on extreme hours,** the opposite of what I expected. Its median error is close to the trees' (2.46 against 2.30), but its worst miss is 424 units, against 115. The likely cause is the conversion back to original units, which grows exponentially in the tails; capping the forecast on the transformed scale would likely prevent it. The combined model inherits this (worst miss 179).
- **2017 was harder than 2016.** The MAE of the trees is 3.08 in 2016 and 3.99 in 2017; for the published linear ensemble it is 2.96 and 4.25.

**Extreme hours and the choice of metric:** In the extreme hours the error of the trees is about 3 times their average, and that of the published ensembles 2.6 to 3.3 times; the trees are still the best of my models there. This follows from the metric: settings were chosen on MAE, for the best average forecast. MAE counts every unit of error the same, while a trading result does not. The next step is therefore a custom metric that measures the monetary gain or loss a forecast error causes. Tuning on it could lead to a different model or different settings, and it matters most if strong outliers are what is traded.