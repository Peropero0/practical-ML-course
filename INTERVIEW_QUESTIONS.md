# Interview questions

One question per exercise, with the answer in short form (click the question to open it). Try to answer out loud before reading.

## Module 1 · Linear regression

<details><summary><b>A1</b> · A colleague computes the regression with (XᵀX)⁻¹Xᵀy and sometimes gets absurd coefficients. Where do you start?</summary>

I check `np.linalg.cond(X.T @ X)`: if it is close to 10¹⁶ the matrix is numerically singular, typically because of collinear features or very different scales. Then I switch to a solver based on QR or SVD (`np.linalg.lstsq`, `LinearRegression`), which does not square the conditioning, I scale the features and, if the collinearity is real, I consider a Ridge.

</details>

<details><summary><b>A2</b> · Why is gradient descent slower (or does it diverge) when the features have very different scales, even though OLS has an exact solution?</summary>

Because the surface of the MSE becomes a very long and narrow valley: the Hessian has very different eigenvalues. The learning rate is limited by the largest eigenvalue (2/λ_max) but the speed along the direction with the smallest eigenvalue is proportional to lr·λ_min, so the ratio λ_max/λ_min (the conditioning) decides how many iterations are needed. Standardizing brings the eigenvalues closer to each other.

</details>

<details><summary><b>A3</b> · In a regression the p-values of the coefficients look great but the residual plot shows a funnel. What do you do?</summary>

The funnel indicates heteroscedasticity: the coefficient estimates remain unbiased, but the classical standard errors are wrong, and so are the p-values and intervals. I would recompute the uncertainty with robust standard errors (HC) or with the pairs bootstrap, and I would check whether a transformation of the target (for example the logarithm) stabilizes the variance.

</details>

<details><summary><b>B1</b> · The model on the logarithm of the price has a better MAPE but a worse RMSE. Which one do you put in production?</summary>

It depends on what the error costs the company. If the relative error matters (for example an estimate shown to the customer, where being off by 30% is bad on both a small and a large house) I choose the model on the logarithm. If the absolute error in euros matters (for example the committed capital) and expensive houses weigh more, I look at the RMSE or the error weighted by value. In any case I would check the bias of the predictions brought back to the original unit.

</details>

<details><summary><b>B2</b> · You used the Lasso to select features and only 6 out of 200 survived. Can you say that the other 194 don't matter?</summary>

No. The Lasso chooses based on that sample: with correlated features it keeps one per group arbitrarily and discards the others, and the selection can change by repeating the fit on another bootstrap sample. I can say that those six are enough, together, to predict well *in that dataset*; not that the others have no relationship with the target. For a substantive conclusion I would check the stability of the selection or use an ElasticNet.

</details>

<details><summary><b>B3</b> · Your degree 10 polynomial model has an excellent test error. Why don't you trust it yet?</summary>

Because the test set has the same distribution as the training set: it measures the ability to interpolate, not to extrapolate. A high-degree polynomial can follow the points well inside the range and diverge right outside it. I would check whether values outside the range will arrive in production (drift), I would look at the predictions on plausible extreme inputs and I would prefer a low-degree model with regularization.

</details>

<details><summary><b>C1</b> · The model has excellent metrics, but the coefficient of a variable has the opposite sign from the expected one. What do you check?</summary>

First the VIFs and the correlation matrix: if the variable has a strongly correlated "twin", the sign is an artifact of the compensation between the two coefficients, not a real effect. I would repeat the estimation on bootstrap samples to see whether the sign is unstable, and then I would merge the two variables, drop one, or use Ridge if I only need to predict. I would not trust the interpretation of the individual coefficients.

</details>

<details><summary><b>C2</b> · Why is an outlier at the margins of the data range more dangerous than one in the center, at the same distance from the line?</summary>

Because it has high leverage: in the hat matrix the weight of an observation on its own prediction grows with the distance of x from the center. A point in the center can be compensated by all the others, a point at the edge cannot: the line bends toward it because few other points constrain it in that area. That is why you look at Cook's distance, which combines residual and leverage.

</details>

<details><summary><b>C3</b> · A colleague reports R² = 0.96 with 80 observations and 70 features. How do you react?</summary>

I ask for the cross-validation before any conclusion: with p close to n the training R² is close to 1 by construction, even on pure noise. I would compute the adjusted R² (or better the cross-validated one) and consider a reduction of the features or a regularization, and I would repeat everything on a held-out test set.

</details>

<details><summary><b>D1</b> · The coefficient of 'years of experience' in your model is 1,051 euros, the one of 'big city' is 5,348. The boss says 'the city matters five times more than experience'. What do you answer?</summary>

That the two numbers are not comparable: 1,051 euros is the effect of **one year** of experience, 5,348 that of the jump from "not a big city" to "big city". To compare importance I would use standardized coefficients or a concrete question: how much does the predicted salary change between an employee at the 10th and one at the 90th percentile of experience, versus changing city? In this dataset experience, over a realistic range, weighs more than the city.

</details>

<details><summary><b>D2</b> · You have a regression on daily sales data and the plot of the residuals against time shows regular waves. Which assumption is violated and what are the consequences?</summary>

The independence of the errors: there is autocorrelation. The coefficient estimates remain reasonable but the standard errors are underestimated, so p-values and intervals are too optimistic. I would confirm with Durbin-Watson, and move to a model that handles the temporal dependence (AR errors, lagged features, seasonal components). For validation I would use TimeSeriesSplit, not a random K-fold.

</details>

<details><summary><b>D3</b> · In your model a single point has a Cook's distance 40 times the threshold. Do you remove it?</summary>

First I understand why: I look at its leverage and its residual, and check whether it is a recording error. If it is an error I correct it or remove it documenting the choice. If it is a real but extreme observation (for example a customer much bigger than the others), I would in effect be removing the part of the range that I want to predict: I prefer a robust model (Huber), a transformation that reduces the leverage, or I state that the model does not cover that area. I always report the results with and without the point.

</details>

<details><summary><b>E1</b> · You have 5000 features and 200 patients. You selected the 20 most correlated with the outcome and then did the cross-validation, with AUC 0.85. Do you trust it?</summary>

No: the selection saw the labels of all the patients, including those of the validation folds, so the AUC is inflated. With p much greater than n it is easy to find 20 features that correlate with the outcome by chance. I would redo the evaluation with the selection inside a Pipeline (or a nested cross-validation), and compare with the AUC obtained by permuting the labels: if that one is also high, the number means nothing.

</details>

<details><summary><b>E2</b> · The model on the logarithm of the price has an R² of 0.94 computed by a script. The direct model has 0.92. Which one do you choose?</summary>

First I check where each R² is computed: if the one of the log model is computed on the logarithm of the price while the other on the price, they are not comparable, they are two different targets. I would recompute both on the same target (the price in euros, after bringing the predictions back with exp) and with the same function, and choose based on the metric that reflects the cost of the error for the business.

</details>

<details><summary><b>E3</b> · In production your model receives one request at a time. How do you apply the scaler to a single request?</summary>

With `transform`, using the mean and standard deviation saved in training: it is also the reason why the scaler object (or better the whole Pipeline) must be serialized with the model. With a single row `fit_transform` would give a standard deviation of zero and would center the value on itself, that is no information.

</details>

<details><summary><b>F1</b> · You have 300 observations, 60 features correlated with each other and you don't know how many matter. Ridge, Lasso or ElasticNet?</summary>

I would not choose a priori: I would compare them in nested cross-validation, because they are three different hypotheses about the world. Ridge assumes that many features contribute a little, Lasso that very few matter, ElasticNet is the compromise and copes better with correlated groups (the Lasso picks one at random per group). With highly correlated features I would start from ElasticNet or Ridge, and I would use the stability of the selection on bootstrap samples if I had to interpret the chosen features.

</details>

<details><summary><b>F2</b> · Why does linear regression not need the scaler but Ridge does?</summary>

OLS minimizes the squared errors without constraining the coefficients: rescaling a feature rescales its coefficient in the opposite direction and the optimum stays the same, so the predictions do not change. Ridge adds α·Σβ²: it is a penalty on the *numeric value* of the coefficients, which depends on the unit of measure. At equal real effect, a feature in small units has a large coefficient and is punished more. The scaler makes the coefficients comparable.

</details>

<details><summary><b>F3</b> · You are predicting delivery times. The distribution is skewed, with long tails. Linear regression or quantile regression?</summary>

It depends on what I have to tell the customer. Linear regression estimates the mean time, pulled up by the tail: a number that is usually higher than what happens in most deliveries. The median describes the typical case better, and it is robust to extreme values. If what matters to the company is not breaking a promise, I would use a high quantile (for example the 90th percentile), which quantile regression estimates directly. I would choose the loss based on the cost of the error.

</details>

<details><summary><b>R1</b> · What does an R² of 0.36 on the test set mean?</summary>

That the model explains 36% of the variance of the target compared with always predicting the training mean (which is worth 0): it is useful but leaves a lot of unexplained variability. The number has to be read on the test set and next to the baseline, and accompanied by an error in the units of the target (RMSE).

</details>

## Module 2 · Preprocessing: scaling, encoding categoricals, missing data

<details><summary><b>A1</b> · In production your service receives one request at a time. How do you apply the scaler to a single row?</summary>

With `transform`, using the mean and standard deviation saved from training: that is why the scaler must be serialized together with the model (better the whole Pipeline). With `fit_transform` on one row the standard deviation would be zero and the result would be meaningless.

</details>

<details><summary><b>A2</b> · A one-hot in production gives a dimension error after a new city shows up. What happened and how do you prevent it?</summary>

Probably the columns were built with `pd.get_dummies` on the incoming data, so the number of columns depends on the categories present: a new city adds a column and the model, trained on a fixed number, refuses. I use an encoder with state (`OneHotEncoder(handle_unknown="ignore")`) inside the Pipeline: the columns are fixed by the training set and the new categories produce zeros. In addition I monitor their frequency as a drift signal.

</details>

<details><summary><b>A3</b> · What is target leakage in target encoding, and how do you avoid it?</summary>

If the encoding of a row uses the mean of the target of its category computed including the row itself, the encoded value contains its target: the smaller the category, the stronger the effect (with a single row, the encoding is the target). You avoid it with out-of-fold encoding (every row encoded with statistics computed without it), with smoothing toward the global mean for rare categories, and by keeping everything inside the cross-validation.

</details>

<details><summary><b>B1</b> · Why do you encode `education` with an OrdinalEncoder and not with a one-hot?</summary>

Because it has a natural order (middle school < ... < phd) and a single column with those codes lets a linear model exploit the progression with a single coefficient, with fewer parameters and less noise. A one-hot ignores the order and adds columns. If I suspected a non-monotonic effect (for example the risk does not grow linearly with the degree), the one-hot would leave the model more freedom.

</details>

<details><summary><b>B2</b> · When would you use imputation with the mean instead of KNN or iterative?</summary>

When the columns are barely correlated (KNN and iterative have nothing to exploit), when the dataset is huge and the computation cost matters, or when I need a robust method that is easy to explain and to put in production. With few missing values the difference in error is negligible. With many missing values and correlated columns I lose information by not using KNN or iterative.

</details>

<details><summary><b>B3</b> · You have a feature with a long tail. Log or QuantileTransformer?</summary>

I start from the model: for trees and boosting nothing is needed (monotonic transformations do not change the splits). For a linear or distance-based model I prefer the log or Yeo-Johnson: they are continuous, monotonic functions that keep working out of range and keep an interpretation (percentage changes). I use the quantile transformer when the distribution is very irregular (bimodal) and I don't expect values outside the range seen.

</details>

<details><summary><b>C1</b> · I standardized the features but the KNN still does badly. What do I check?</summary>

That the irrelevant features are not diluting the distance: with many barely informative columns the neighbors become almost equidistant (curse of dimensionality). I try feature selection, a dimensionality reduction (PCA), the metric (Manhattan is less sensitive) or weighting the features. And I check that k is chosen in cross-validation.

</details>

<details><summary><b>C2</b> · When is the OrdinalEncoder acceptable on a nominal variable?</summary>

With tree models (trees, forests, boosting): the model works with thresholds and can isolate a category with two splits, so it is not bound to the order. With few categories and enough data it works; with many categories you need many splits and a better encoding is preferable (target encoding, native handling of categoricals). With linear models, KNN, SVM or networks it is wrong: you need the one-hot or another encoding.

</details>

<details><summary><b>C3</b> · Why is imputation with the mean criticized in statistical analyses, but often used in machine learning?</summary>

Because in statistical analysis what matters are variances, correlations and coefficients, and the mean distorts them systematically (it shortens the variance by a factor 1 − p and weakens the correlations). In machine learning the goal is predictive performance: with few missing values and a flexible model the distortion costs little, and the mean is simple, fast and robust. In both cases I add the missingness indicator and check the effect with cross-validation.

</details>

<details><summary><b>D1</b> · What is the dummy variable trap and when is it a problem in scikit-learn?</summary>

It is the linear dependence between the one-hot columns of a feature and the intercept: their sum is the column of ones. With an unregularized OLS the matrix is not invertible and the coefficients are not identifiable; you avoid it by removing one column (`drop="first"`). In scikit-learn the regularized models (Ridge, `LogisticRegression` with L2, the default) work anyway, so often nothing is dropped. If I want to interpret the coefficients as differences from a baseline chosen deliberately, I drop that category myself.

</details>

<details><summary><b>D2</b> · A colleague says: 'I checked, the missing data are MCAR, so I drop the rows'. How do you reply?</summary>

That MCAR cannot be verified: the tests can only detect dependencies on the observed variables, and a non-significant result is not proof (it depends on the power). Besides, MNAR is indistinguishable from the observed ones. Dropping the rows with many missing values also loses data and can introduce bias. I would compare dropping, simple imputation and imputation with indicator in cross-validation on the real problem, and use knowledge of how the data are collected.

</details>

<details><summary><b>E1</b> · Why do you use `OneHotEncoder` and not `pd.get_dummies` in a production pipeline?</summary>

Because the former is a transformer with state: the categories are learned from the training set and saved together with the model, so the columns are always the same and new categories have a defined behavior (zeros, or an error if I want one). `pd.get_dummies` is a function with no memory: the result depends on the data I pass it, and with a single request it would produce a single column. Inside a Pipeline, moreover, it guarantees that training and production use the same code.

</details>

<details><summary><b>E2</b> · How do you get the feature names after a Pipeline with ColumnTransformer and OneHotEncoder?</summary>

With `get_feature_names_out()` on the preprocessor (`pipe[:-1].get_feature_names_out()` or `pipe.named_steps['prep']...`), which returns the names in the order of the output columns, with the step prefix (`num__income`, `cat__city_Rome`). I use them to label coefficients or importances; I never rebuild them by hand, because the order depends on the blocks and on the number of categories.

</details>

<details><summary><b>E3</b> · Why must a transformer that uses the target be written with a `fit_transform` different from `fit` + `transform`?</summary>

Because in training the encoding of every row must not contain its target: `fit_transform` computes it out-of-fold. In validation or in production, instead, the rows are new and the mapping of the whole training set is used (`transform`). If `fit_transform` were `fit` + `transform` on the same training set, the model would see a feature that has already seen the target and would learn to trust it too much: in production it will not be like that.

</details>

<details><summary><b>F1</b> · You have a dataset with outliers and a KNN. Which preprocessing do you propose, and how would you validate it?</summary>

I would start from a quantile-based scaler (`RobustScaler`, or `QuantileTransformer` if the distribution is very irregular), because minimum/maximum and mean/standard deviation get dragged by the outliers. I would compare it with `StandardScaler` in cross-validation inside a Pipeline, for the final model, and I would also check whether the outliers are errors (to correct) or real cases (to keep).

</details>

<details><summary><b>F2</b> · For a variable with 500 categories and a gradient boosting model, which encoding do you use?</summary>

I would start from the native handling of categoricals (LightGBM, CatBoost, or `HistGradientBoostingClassifier` with `categorical_features`), which avoids the explosion of columns and does not require choosing an order. Alternatively out-of-fold target encoding with smoothing, or frequency. One-hot with 500 columns slows things down and fragments the data; ordinal works with trees but with no quality guarantees. I would compare the options in cross-validation inside the pipeline.

</details>

## Module 3 · Pipelines, cross-validation and hyperparameter search

<details><summary><b>A1</b> · What happens, step by step, when you call `cross_val_score(pipeline, X, y, cv=5)`?</summary>

At every fold `cross_val_score` clones the pipeline (parameters, not state), trains it on the training part of the fold (`fit`: the intermediate steps do `fit_transform`, the last one `fit`) and evaluates it on the validation fold (`predict`: only `transform` and then `predict`). The preprocessing statistics are therefore recomputed 5 times, each only on the training data of the fold: it is the reason why there is no leakage.

</details>

<details><summary><b>A2</b> · Why must the model be cloned and not reused from one fold to the next in cross-validation?</summary>

Because `fit` does not reset the state of models that train incrementally (initial coefficients with `warm_start`, some models with `partial_fit`), and even where it does reset it the cost of getting it wrong is silent. Reusing a model already trained on other folds contaminates the evaluation with information from the past. `clone` creates a model with the same hyperparameters and no state.

</details>

<details><summary><b>A3</b> · How many trainings does GridSearchCV do with 3 hyperparameters of 4 values each and 5 folds?</summary>

4 × 4 × 4 = 64 combinations, for 5 folds: 320 trainings, plus one refit on the full training set: 321. It is the reason why with many hyperparameters you move to RandomizedSearchCV or to methods that discard poor combinations early (halving).

</details>

<details><summary><b>B1</b> · Why do imputation and scaling sit inside a Pipeline and not in a separate step before the split?</summary>

Because they are steps that learn from the data (median, mean, standard deviation): if I run them before the split, the test set contributes to the statistics. Inside a Pipeline every cross-validation fold refits them on the training part of the fold alone, and in production the same object applies exactly the statistics learned during training.

</details>

<details><summary><b>B2</b> · How do you fairly compare a random forest and a logistic regression on a dataset with common preprocessing?</summary>

With the same preprocessing pipeline, the same folds and a single `GridSearchCV` with a list of grids (a placeholder step for the model), so every family has its own hyperparameters optimized with the same cross-validation budget. Then I evaluate the best on the test set once, and look at whether the difference from the second exceeds the variability across folds.

</details>

<details><summary><b>B3</b> · What do you save to put a model in production, and what can break on loading?</summary>

I save the whole pipeline (preprocessing and model) with `joblib`, together with the version of Python and of the libraries and a small set of example requests with their predictions, to use as a regression test on loading. It can break because of the scikit-learn version (the internal format changes), because of custom functions not importable in the service, or because of input columns different from those of the training set.

</details>

<details><summary><b>C1</b> · Your sales forecasting model has R² 0.95 in cross-validation but in production it is very wrong. Where do you start?</summary>

From how I validated: if I used a random K-fold on temporal data, the validation is an interpolation, not a forecast of the future. I redo the evaluation with `TimeSeriesSplit` (or a chronological hold-out), compare with naive baselines (last value, same period of the previous year) and check that no feature contains information from the future. If the model does not beat the baseline, it is simply not ready.

</details>

<details><summary><b>C2</b> · You have 5,000 transactions from 200 customers and you want to evaluate a fraud model that will have to work on new customers. How do you validate?</summary>

With `GroupKFold` (or `StratifiedGroupKFold` if the positive class is rare) grouping by customer, so that all the transactions of a customer are in the same fold. A random K-fold would put transactions of the same customer in training and in validation and the model could recognize the customer, inflating the estimate. I also check that the estimate with the groups is consistent with what I see on truly new customers in production.

</details>

<details><summary><b>C3</b> · When do you really need a nested cross-validation, and when is a test set enough?</summary>

It is needed when the data are few and I cannot afford a separate test set of sufficient size, or when I have to compare different tuning procedures and report an unbiased estimate (for example in a publication or in a regulatory review). With plenty of data a test set held out and touched once is enough: it costs much less and does not require repeating the search in every fold.

</details>

<details><summary><b>D1</b> · Your GridSearchCV says the best has AUC 0.812 with max_depth=8, but the second has 0.811 with max_depth=4. Which one do you send to production?</summary>

I look at the standard deviation across folds: if it is 0.03 the difference of 0.001 is noise, and I apply the standard error rule choosing the simplest model among the indistinguishable ones: the one with max_depth=4 (less overfitting, faster, more stable). If I had reasons to prefer the other (for example a different test set that confirms it) I would write them down, but "first in the ranking" is not a reason.

</details>

<details><summary><b>D2</b> · Your learning curve shows a large and stable gap between training and validation, even with all the data. What do you do?</summary>

A large and stable gap is overfitting that additional data won't solve by themselves: first I reduce the complexity of the model (regularization, less depth, fewer features, early stopping), then I reevaluate. Only if the gap shrinks as the data grows (the validation curve still rises) does it make sense to collect more. If training and validation are both low and close, instead, the problem is the model being too simple or the features being barely informative.

</details>

<details><summary><b>E1</b> · After evaluating on the test set, can I retrain the model on training + test before putting it in production?</summary>

Yes, but the score to report is the test set score obtained **before** retraining: it is the honest estimate of the chosen model. The final model trained on more data will probably be a bit better, but I no longer have an independent set to measure it. If I have to change something after seeing the test set, the test set is "burned" and I need a new holdout set.

</details>

<details><summary><b>E2</b> · How do you notice that a Pipeline with ColumnTransformer is ignoring an important column?</summary>

I look at the output names (`get_feature_names_out()`) and `transformers_`: the `remainder` entry with non-empty indices says which columns were dropped. In the training code I add a check that fails if the remainder is not empty, and in the service I verify the input columns against `feature_names_in_`. An AUC lower than expected with no errors is often the first signal.

</details>

<details><summary><b>E3</b> · When is it legitimate to use `error_score=np.nan` in a hyperparameter search?</summary>

When some combinations can legitimately fail (for example a solver that does not converge or an incompatible combination of parameters) and I don't want to interrupt the whole search. The failed combinations get NaN and are excluded from the ranking. But I always check how many failed (`cv_results_`) and investigate: if they all fail, or one particular fold fails, the error has a cause (data, splitter) that I am hiding.

</details>

<details><summary><b>F1</b> · Why 5 or 10 folds and not leave-one-out, which uses more data for training?</summary>

Because leave-one-out has the lowest bias but not the lowest variance: the 200 models are almost identical (they differ by one row) and the errors are highly correlated, so the mean has less information than it seems; moreover it costs n trainings. With 5 or 10 folds the compromise between bias (slightly smaller training set) and variance and cost is better, and with repetition the variance goes down further.

</details>

<details><summary><b>F2</b> · You have a budget of 100 trainings and 6 hyperparameters. Grid or random search?</summary>

Random search: with 6 hyperparameters a grid of 100 combinations would have about 2 values per parameter, that is almost no exploration, while 100 random samples test 100 different values of every parameter. If a few parameters matter much more than the others (as is almost always the case) random search finds the good region with the same budget. Then I may refine with a narrow grid around the best, or use a method with early discarding like halving.

</details>

## Module 4 · Logistic regression and classification metrics

<details><summary><b>A1</b> · Why the log-loss and not the squared error for logistic regression?</summary>

Because with the sigmoid the squared error is not convex (several local minima) and has tiny gradients when the model is sure and wrong, so it learns slowly exactly where it matters. The log-loss is the negative log-likelihood of a Bernoulli model (maximum likelihood estimation): it is convex, its gradient is simply `(p - y) x` and it heavily penalizes whoever is sure and wrong.

</details>

<details><summary><b>A2</b> · How do you interpret an AUC of 0.78?</summary>

If I pick a defaulting customer and a non-defaulting one at random, in 78% of the cases the model gives the higher score to the defaulter. It says nothing about the calibration of the probabilities nor about which threshold to use: it measures only the ordering. That is why with very imbalanced classes I pair it with precision and recall at the chosen threshold or with the average precision.

</details>

<details><summary><b>B1</b> · You have columns with missing values: how do you decide how to treat them?</summary>

First I understand **why** they are missing (random, linked to another column, linked to the target). If the mechanism is informative I add a missingness indicator on top of a simple imputation (median or most frequent) inside the pipeline, and measure with an ablation in cross-validation how much it is worth. More sophisticated imputations (KNN, iterative) are justified only if they really gain over the median plus the indicator.

</details>

<details><summary><b>B2</b> · How do you choose the threshold of a classifier in production?</summary>

I start from the costs (or from the operational constraint: for example "I can check at most 100 cases a day"), not from the metric. With calibrated probabilities the theoretical threshold is c_FP/(c_FP+c_FN); in practice I search for it on a grid with out-of-sample predictions on the training set and validate it once on the test set. I recheck it when the prevalence or the costs change, because the threshold depends on both.

</details>

<details><summary><b>B3</b> · What is the difference between macro, micro and weighted average?</summary>

Macro: simple mean of the per-class metrics, every class weighs the same (sensitive to rare classes). Micro: counts TP, FP, FN over all the classes together (for single-label multiclass it coincides with the accuracy). Weighted: mean weighted by the number of examples of each class. I use macro when the rare classes matter, micro or accuracy when the total is what matters.

</details>

<details><summary><b>C1</b> · What happens to an unregularized logistic regression if the classes are linearly separable?</summary>

The likelihood has no finite maximum: as the norm of the weights increases the loss keeps decreasing toward zero, so the coefficients diverge and the solver stops only because of tolerance or the maximum number of iterations. The probabilities become 0 or 1, the standard errors of the coefficients are nonsensical. It is solved with a penalty (L2, or L1 if I also want to select) or with a Bayesian regularization, and the strength is chosen in cross-validation.

</details>

<details><summary><b>C2</b> · Your fraud model has AUC 0.92 but the antifraud team says 95% of the alarms are false. Is the model wrong?</summary>

Probably not: with a prevalence of 0.5% even an excellent classifier has low precision, because the precision depends on the prevalence (P = R·π / (R·π + FPR·(1−π))). I check the precision on the real stream, compare it with the formula, and set the threshold based on how many alarms the team can work and the cost of an undetected fraud. The AUC says the ordering is good; the precision says how much it costs to act on the top of the list.

</details>

<details><summary><b>D1</b> · Two highly correlated variables have coefficients of opposite sign in your regression. How do you explain it to the team?</summary>

I say the coefficient measures the effect of one variable holding the other fixed, and when they are correlated the common part is split arbitrarily: their differences matter, not the single values. I check the collinearity (VIF), try the model with only one at a time and with a constructed feature (a difference, a ratio), look for omitted variables that explain the opposite sign, and if the goal is prediction I don't interpret the individual coefficients. For a causal effect you would need a different design, not a better model.

</details>

<details><summary><b>D2</b> · Your model says 0.3: does it mean that 30% of these cases are positive?</summary>

Only if the model is calibrated. I verify it with a calibration curve (observed frequency against predicted probability for groups of cases) and an indicator like the ECE or the Brier score, on data not used for training. The logistic is usually calibrated; forests, boosting and SVMs are not, and neither are models trained with weights or on rebalanced data: in that case I recalibrate (sigmoid or isotonic) with `CalibratedClassifierCV`.

</details>

<details><summary><b>D3</b> · You have a macro F1 much lower than the accuracy. What does it tell you?</summary>

That there are (rare) classes on which the model does badly and that the accuracy hides them because they count little in the total. I look at precision and recall per class and the confusion matrix to see where the errors go, then I evaluate interventions: more data or weights for the rare classes, features that tell them apart, different thresholds per class if the decisions allow it.

</details>

<details><summary><b>E1</b> · Which models require feature scaling and which don't?</summary>

Those that use distances (KNN, kernel SVM, k-means), PCA and the **regularized** linear ones or those trained by gradient descent (logistic, Ridge, Lasso, networks). Trees and their ensembles (random forest, boosting) use thresholds and are invariant to monotonic transformations of the individual features, so they don't need it. When in doubt I run an invariance test: I change the unit of a column and see whether the predictions change.

</details>

<details><summary><b>E2</b> · How do you make sure you use the right column of predict_proba?</summary>

I encode the target as 0/1 with the class of interest as 1 at the start, and check `model.classes_`. If the target has text labels I look up the column by name and not by position. In the automated tests I verify that the mean predicted probability is close to the frequency of the positive class: it is a very simple check that immediately catches swapped columns.

</details>

<details><summary><b>E3</b> · Why must SMOTE be applied after the split and inside the cross-validation?</summary>

SMOTE creates synthetic examples by interpolating between nearby positives: if I apply it before the split, a synthetic example in the training set is almost identical to the real positive that ends up in the test set, so the model "recognizes" it. You need a pipeline that applies the sampling only on the training part of every fold (with `imblearn.pipeline.Pipeline`), and the test/validation must keep the real distribution.

</details>

<details><summary><b>F1</b> · When would you use L1 instead of L2 for a logistic?</summary>

When I think few variables really matter and I want a sparse, interpretable model, or when I have many more features than examples. L2 when the signal is diffuse or the features are highly correlated (L1 picks one of the group unstably). Elastic net is the compromise. In all cases I standardize the features and choose `C` (and `l1_ratio`) in cross-validation, remembering that the selection done by L1 must be checked for stability.

</details>

<details><summary><b>F2</b> · You have a dataset with 1% positives. Where do you start?</summary>

From the base model without rebalancing, evaluated with average precision and precision-recall curves (not accuracy), and from the threshold chosen on the costs or in cross-validation. Only if the results are insufficient do I try class weights or sampling, always inside the validation (only on the training part of every fold) and comparing the probabilities and the calibration, not just the recall. With tree models I also consider the hyperparameters that control depth and minimum leaves, which often matter more than rebalancing.

</details>

<details><summary><b>R4</b> · Why does lowering the threshold increase the recall and reduce the precision?</summary>

Lowering the threshold classifies more cases as positive: more of the true positives are caught (recall up) but more false positives come in too (precision down). The threshold is chosen with the costs of the two errors and validated on data not used to choose it.

</details>

## Module 5 · K-nearest neighbors

<details><summary><b>A1</b> · What is the complexity of `predict` for a KNN and how is it reduced?</summary>

For every test point it is O(n·d) with the exhaustive computation (n training examples, d features), plus O(n) to select the k minima with a partition. It is reduced with search structures (kd-tree, ball-tree: good with few dimensions) or with approximate indexes (LSH, HNSW) in high dimension, at the cost of some errors in the neighbors found. In production you also precompute the scalers and keep the training set in memory.

</details>

<details><summary><b>A2</b> · Why is leave-one-out cross-validation cheap for KNN?</summary>

Because KNN has no training: the prediction for a point uses its neighbors in the training set excluding the point itself, so a single distance matrix with the diagonal excluded is enough. With a single sort per row you get the predictions for every `k` at almost no cost. What remains is the O(n²) cost of the matrix, and the variance of the LOO estimate (the n estimates are highly correlated), which is why on large datasets I prefer a repeated K-fold.

</details>

<details><summary><b>B1</b> · How do you treat categorical variables in a distance-based model?</summary>

One-hot with `handle_unknown` is the starting point, but with many categories or few informative ones it adds noise to the distance: I check with an ablation in cross-validation, consider weighting the block (multiplying the dummies by a factor), reducing the levels or using a distance designed for mixed data (Gower). If the categories matter a lot, a tree model handles them better and does not require these choices.

</details>

<details><summary><b>B2</b> · How do you decide which features to use in a KNN and with what weight?</summary>

I start from a few features that have a proximity meaning for the problem (coordinates, time, product categories), scale them and try a weight per feature (or group of features) as a hyperparameter, evaluating on a validation set or in cross-validation. If the candidate features are many, I use a selection or a reduction (PCA, metric learning) before the KNN, or I switch to a model that learns by itself which features matter (trees, regularized linear models).

</details>

<details><summary><b>B3</b> · How do you evaluate an imputation method?</summary>

With the known-truth trick: I hide a share of values I know (with the missingness mechanism I consider realistic), impute and measure the error on the hidden cells, besides the effect on the final task in cross-validation with the imputer inside the pipeline. I remember that the mechanism matters: if the data are missing in a non-random way, the results on random missingness are optimistic.

</details>

<details><summary><b>C1</b> · What is the curse of dimensionality and how does it hit KNN?</summary>

As the number of dimensions grows the space "empties out": the points are almost all at the same distance from each other, the ratio between the nearest neighbor and the farthest tends to 1 and the neighbors stop being meaningful. In KNN it is a direct problem because the prediction depends only on the distance. It is mitigated with feature selection or reduction, with learned metrics or with models that select them by themselves (trees, regularized linear models), or with a lot more data.

</details>

<details><summary><b>C2</b> · You suspect that 20% of the labels are wrong. What do you change in a KNN?</summary>

I increase `k` (and choose it in cross-validation, which sees the same noise) to compensate for the errors by voting among more neighbors; I use 1/distance weights with caution because a wrong neighbor very close weighs too much. If I can, I clean up: methods like `k=1` with a filter (edited nearest neighbours) or the search for labels inconsistent with the neighbors. And I always measure on a test set with verified labels, otherwise I measure the noise too.

</details>

<details><summary><b>D1</b> · How do you explain to a colleague that `k` regulates the complexity?</summary>

With `k = 1` the prediction at a point depends only on the nearest neighbor: the boundary follows every example and every error, it has dozens of regions and the training is perfect. As `k` grows the prediction is an average over more points, the boundary simplifies and in the end it becomes the most frequent class. It moves the model along the bias-variance tradeoff; I choose it with cross-validation.

</details>

<details><summary><b>D2</b> · Can you give a probability with a KNN?</summary>

Yes, but with caution: the fraction of neighbors is a coarse, discrete estimate, poorly calibrated with small `k`. I use a larger `k`, check the calibration with a curve or an indicator like the ECE and, if needed, recalibrate it (sigmoid or isotonic) on data not used for training. If the probabilities are central to the use case, I consider a model that estimates them directly, like the logistic.

</details>

<details><summary><b>E1</b> · How do you do stacking (or target encoding) without leakage?</summary>

The features built from the target must be computed **out of sample**: for every row of the training set I use a model or statistic trained on the other rows (K-fold: `cross_val_predict`, target encoding with folds). For the test set I use the model trained on the whole training set. That way the distribution of the feature in training and in test coincides and the next model learns to trust it by the right amount.

</details>

<details><summary><b>E2</b> · How do you avoid schema errors when a model goes into production?</summary>

I train on DataFrames with selection of the columns by name inside the pipeline, save the list of expected columns with the model and verify the incoming schema (names, types, ranges). I add automated tests with example data in a different order and with missing columns, and monitor the distribution of the features in production against that of the training set.

</details>

<details><summary><b>E3</b> · What is the difference between StandardScaler, MinMaxScaler and Normalizer?</summary>

The first two work per **column** with statistics learned from the training set: StandardScaler removes the mean and divides by the standard deviation, MinMaxScaler brings every column into an interval (usually 0-1). Normalizer works per **row** and divides every example by its norm: it is suited when the direction matters (text, cosine) and does not change the relative scales of the columns. I choose based on what must be comparable: the columns with each other or the examples with each other.

</details>

<details><summary><b>F1</b> · Do I always have to scale the features before a KNN?</summary>

Almost always, because the distance sums the differences and different scales decide by themselves. But it is not a rule: if the features already have the same meaning and the same scale (pixels, histograms, normalized counts) standardizing can amplify the noise of the nearly constant columns. I always compare with and without a scaler in cross-validation and also look at the metric, which usually counts less.

</details>

<details><summary><b>F2</b> · Tabular data, 50 features, 200,000 rows, classification goal: where do you start?</summary>

From a linear baseline (regularized logistic) and a tree model (random forest or gradient boosting): they are robust, scale well and at dimension 50 the KNN would have problems both of distance and of prediction cost. I compare with cross-validation, look at the calibration and the required latency. I reserve KNN for problems where proximity has a direct meaning (recommendations, similarity search, geographic data) and in that case with approximate indexes.

</details>

<details><summary><b>R5</b> · Why does KNN need standardization and a tree does not?</summary>

KNN measures distances: a variable with a large scale dominates the computation. A tree uses thresholds on one variable at a time, so changing the units moves the thresholds but not the order of the examples and the result does not change.

</details>

## Module 6 · PCA: principal component analysis

<details><summary><b>A1</b> · How is PCA computed and what is the relationship with the eigenvalues?</summary>

You center the data and compute the SVD of the centered matrix X = U S V^T: the rows of V^T are the components and S^2/(n-1) are the variances. It is equivalent to the spectral decomposition of the covariance (the eigenvectors are the components, the eigenvalues the variances), but the SVD is numerically more stable and for large matrices truncated or randomized variants are used. The projection onto the first k components is the best rank-k approximation in squared error.

</details>

<details><summary><b>A2</b> · How do you compute PCA on a matrix with millions of rows and thousands of columns?</summary>

You don't need the complete decomposition: the first k components are enough. You use iterative or randomized methods (power/subspace iteration, Lanczos, randomized SVD) that work on the matrix with matrix-vector products, or `IncrementalPCA` in blocks if the data don't fit in memory. The speed depends on the gap between eigenvalues: with small gaps you need more iterations or oversampling.

</details>

<details><summary><b>B1</b> · How do you choose the number of principal components?</summary>

It depends on the use. For visualization 2-3. For reconstruction a threshold on the cumulative variance or the required reconstruction error. For a downstream model I treat `k` as a hyperparameter and search it in cross-validation inside the pipeline, with a parsimony rule (the smallest k within the standard error of the best). I remember that the explained variance measures reconstruction fidelity, not usefulness for the target.

</details>

<details><summary><b>B2</b> · How can you use PCA for denoising and how do you choose how many components to keep?</summary>

I reconstruct the data with the first k components, which contain the correlated signal while the noise, spread over all the directions, is mostly discarded. I choose k with a clean reference if I have one, or by comparing the eigenvalues with the expected noise level (Marchenko-Pastur edge), or with cross-validation on a downstream task. It works if the signal is low-dimensional and the noise is uncorrelated; with correlated noise or a nonlinear signal you need something else (autoencoder, kernel PCA).

</details>

<details><summary><b>B3</b> · You have many correlated variables in a regression. What options do you have?</summary>

Ridge (shrinks the coefficients, a single hyperparameter), Lasso or elastic net if I want to select, PCR (regression on the first components) if I think the signal sits in the high-variance directions, PLS if I want target-oriented components. I compare them in cross-validation with the same split; Ridge is usually the starting point because it is stable and requires no discrete choices.

</details>

<details><summary><b>C1</b> · Why does PCA not guarantee that the chosen components are useful for predicting?</summary>

Because it is unsupervised: it orders the directions by variance of X and not by relationship with y. If the signal sits in a low-variance direction (a difference between two large, correlated variables, for example) it gets discarded first. That is why `k` must be chosen with cross-validation on the final task, or you use supervised methods (PLS, LDA) or models that see the original features (Ridge, trees).

</details>

<details><summary><b>C2</b> · Is PCA sensitive to outliers?</summary>

Very: it is based on variance and covariance, that is on squares, so a few distant points can hijack the first component. I inspect the outliers first (Mahalanobis, isolation forest), clip or remove them on the training set alone, or use a robust version (MCD covariance, robust PCA). I always check that the first components are stable with a bootstrap or by removing a few points.

</details>

<details><summary><b>D1</b> · How do you interpret the principal components?</summary>

I look at the explained variance (scree plot) and at the loadings: which variables have a high weight and with which sign, so I see which ones move together. I verify the relationship with external variables (for example the correlation of the scores with the target), check the stability (bootstrap) and don't read the sign in absolute terms. I remember that a component is a direction of maximum variance and not a causal factor.

</details>

<details><summary><b>D2</b> · A 2D PCA plot shows overlapping classes. What do you conclude?</summary>

Little: two components capture a fraction of the variance (here 29%) and the overlap can be an artifact of the projection. I measure the separability with a classifier in more dimensions (cross-validation) and use a nonlinear technique (t-SNE, UMAP) only to explore, remembering that the distances in those maps are not quantitative.

</details>

<details><summary><b>E1</b> · Why can't you do fit_transform on the test set too?</summary>

Because fit learns parameters from the data (means, deviations, components) and the downstream model was trained with the parameters of the training set. Redoing the fit on the test set would change the frame of reference (with PCA: different axes and signs), so the scores would not be comparable with the ones the model saw. You use transform with the objects already trained, and a Pipeline guarantees it by construction.

</details>

<details><summary><b>E2</b> · What is the difference between PCA and TruncatedSVD?</summary>

PCA centers the data and then does the SVD, so it finds the directions of maximum variance around the mean. TruncatedSVD does the SVD directly on the matrix, without centering: it is suited to sparse matrices (where centering would destroy the sparsity, as in LSA for text), but on dense data with a nonzero mean the first component aligns with the mean. If the data are already centered they coincide.

</details>

<details><summary><b>E3</b> · Why do you standardize before a PCA?</summary>

Because PCA looks for the directions of maximum variance, and the variance depends on the units of measure: a column expressed in small units (hence with large numbers) dominates the components. By standardizing every column has variance 1 and PCA works on the correlation matrix, that is on the relationships between the variables. If the columns already have the same unit and the differences in variance are informative (pixels), I don't standardize.

</details>

<details><summary><b>F1</b> · PCA or feature selection: how do you choose?</summary>

It depends on where I think the signal is. If the columns are many and redundant (correlated sensors, pixels) PCA compresses well without losing anything. If a few predictors matter and the rest is noise, PCA mixes them with the noise and supervised selection (filters, L1, trees) works better. If the signal is in low-variance directions, PCA throws it away. In practice I try both in a pipeline with cross-validation and compare them with the no-reduction baseline.

</details>

<details><summary><b>F2</b> · When is linear PCA not enough?</summary>

When the structure of the data is nonlinear (curved manifolds, nested clusters): two concentric circles, images with deformations. The options are KernelPCA (RBF kernel), Isomap or embedding methods (t-SNE, UMAP) for visualization, autoencoders for compression. Kernel methods cost O(n^2) in memory and time, so they are used with small datasets or with approximations (Nystroem). And before complicating things, I check whether a nonlinear model directly on the original data already solves the problem.

</details>

<details><summary><b>R6</b> · What does it mean that the first two components explain 55% of the variance?</summary>

That projecting the data onto the plane of the first two components preserves 55% of the total variance (of the standardized variables): it is a measure of reconstruction fidelity, not of usefulness for predicting something. It is looked at together with the ability of the model to use the components.

</details>

## Module 7 · Decision trees

<details><summary><b>A1</b> · How does a decision tree choose the split and what is the cost?</summary>

For each feature it sorts the examples and tries all the thresholds between consecutive different values, computing the impurity (Gini or entropy) of the two children with cumulative counts; it picks the feature-threshold pair with the maximum gain and repeats recursively on the children until the stopping criterion. The cost per node is dominated by the sorting, O(p n log n); overall it is O(p n log n) per level of the tree. It is a greedy algorithm: it does not guarantee the optimal tree.

</details>

<details><summary><b>A2</b> · How is feature importance computed in a tree?</summary>

For each feature I sum the impurity reduction of the splits that use it, weighted by the fraction of examples that go through the node, and normalize to sum 1 (in forests you average over the trees). It is fast but it has limits: it is computed on the training set, it favors features with many distinct values and it is split among correlated features. For a more reliable judgment I use the permutation importance on validation data.

</details>

<details><summary><b>B1</b> · A decision tree is interpretable: how far do you trust its rules?</summary>

The rules describe precisely how that model decides, but a single tree is unstable: small changes in the data can change the root and the thresholds. That is why I regularize it (depth or minimum leaves chosen in cross-validation), check the stability with a bootstrap and use the rules as an explanation of the model, not as a discovery about the world. If accuracy is needed I use a forest or a boosting and explain with permutation importances and SHAP.

</details>

<details><summary><b>B2</b> · How do you regularize a regression tree?</summary>

With the minimum leaf size (`min_samples_leaf`), which is the most direct control of the variance, with the maximum depth and with cost-complexity pruning (`ccp_alpha`), choosing in cross-validation. An unregularized tree memorizes the training set: here it was worth as much as a straight line, with the right minimum leaf size it beat the line by over 15%.

</details>

<details><summary><b>B3</b> · What is cost-complexity pruning?</summary>

I build the full tree and then minimize error + alpha·(number of leaves): for each alpha there is an optimal subtree and the subtrees are nested, so you get a sequence of smaller and smaller trees. I choose alpha in cross-validation, usually with the standard error rule to prefer the simplest tree. Unlike stopping the growth early, it looks at the overall effect of a branch.

</details>

<details><summary><b>C1</b> · Why is a random forest not suited to predicting a series with a trend?</summary>

Because trees predict means of values seen in training: outside the range of the data the prediction is flat, so it cannot continue a trend. You remove the trend (linear regression, differencing), model the rest with the tree and put it back together; or you use time series models. And you validate with a temporal split, which shows the problem right away.

</details>

<details><summary><b>C2</b> · Why does a tree have high variance and how do you reduce it?</summary>

Because every split depends on few examples and the choices cascade: a small change in the data moves a threshold and changes the whole subtree. You reduce it by regularizing (minimum leaf size, depth, pruning) or by averaging many trees trained on different samples (bagging, random forest): the average of estimators that are little biased but very variable has a much smaller variance, without increasing the bias.

</details>

<details><summary><b>D1</b> · How do you explain the prediction of a tree to a customer?</summary>

I show the path from the example to the leaf (`decision_path`) as a list of conditions ("male, third class, age above 9"), and the fraction of similar training examples that are in the leaf, with their number: it is an exact local explanation, not an approximation. With deep trees or forests the explanation becomes more complex and I use permutation importances or SHAP.

</details>

<details><summary><b>D2</b> · Why can the feature_importances_ of a random forest be misleading?</summary>

Because it is computed on the training set and with the impurity: features with many distinct values (continuous, identifiers) have more candidate thresholds and more chances of reducing the impurity by chance, so they receive importance even if they are noise. I use the permutation importance on a validation set, which measures how much the performance gets worse by shuffling the column, or SHAP; and I keep in mind that correlated features share the importance.

</details>

<details><summary><b>E1</b> · How do you assign names to the importances of a model inside a pipeline?</summary>

With `get_feature_names_out()` of the preprocessing part (`pipeline[:-1]`), because the final model sees columns that are reordered, expanded or removed. Then I check with a second method, like the permutation on the input columns, that the conclusions agree, and I add an automatic test that compares the length and the names.

</details>

<details><summary><b>E2</b> · Your model has a very high AUC in validation. What do you check before trusting it?</summary>

The feature importances (a dominant column, especially if it is an id or a timestamp), the structure of the validation (same period/same customers in training and test), consistency with a simple baseline, and I run a test with new data obtained the way it would be in production. I remove identifiers and columns that encode the collection process, and repeat the evaluation with a temporal or group split.

</details>

<details><summary><b>E3</b> · Why can a hyperparameter found on one dataset fail on another, even a similar one?</summary>

Because many depend on the scale of the data or on the sample size: the ccp_alpha of a regression tree is in units of the target squared, the regularization of a logistic regression depends on the scale of the features and on n, the learning rate on the number of trees. I always search the hyperparameters on the current problem, and when I transfer a value I re-express it in dimensionless form (for example as a fraction of the variance of the target).

</details>

<details><summary><b>F1</b> · Which criterion do you use to split a tree: Gini or entropy?</summary>

In practice they give almost identical trees: I pick the default (Gini, slightly faster) and focus on what really matters, which is the regularization (minimum leaf size, depth, pruning) chosen in cross-validation. If I want to compare the two criteria I do it with a repeated validation and looking at the variability.

</details>

<details><summary><b>F2</b> · When does a tree work better than a logistic regression?</summary>

When the relationship with the target is made of thresholds and interactions (effects that exist only for certain ranges of several variables) rather than smooth linear combinations: XOR, "if A and not B" rules, step effects. With oblique boundaries or many weakly informative features the regularized logistic is often better and more stable. Another advantage of trees is the invariance to monotonic transformations of the features: no scaling. I compare them in cross-validation on the same metric.

</details>

<details><summary><b>R7</b> · Why does a tree with no depth limit have accuracy 1.0 on the training set?</summary>

Because it keeps splitting until every leaf is pure: it memorizes the training set. It has low bias and high variance, and on the test set it generalizes like a much smaller tree. It is controlled with the depth, the minimum leaf size or cost-complexity pruning.

</details>

## Module 8 · Random forest

<details><summary><b>A1</b> · What is the out-of-bag error and when can you trust it?</summary>

Each tree is trained on a bootstrap that leaves out about 37% of the examples: predicting each example only with the trees that have not seen it gives me an honest prediction for the whole training set, and I compute its error, without a validation set. It is a good estimate when the examples are independent; with duplicate or correlated rows (same customer, time series) it is optimistic and you need a validation by group or by time.

</details>

<details><summary><b>A2</b> · Why does a random forest work better than bagging?</summary>

Because at every split it limits the candidate features: the trees end up less correlated (they do not all choose the same dominant feature) and the variance of the average is rho·σ² + (1−rho)·σ²/B, where extra trees alone do not reduce the correlation term. Each tree is a little worse (more bias), but the average of less correlated trees gains much more. It is the trade-off that `max_features` controls.

</details>

<details><summary><b>B1</b> · How do you choose the hyperparameters of a random forest without cross-validation?</summary>

With the out-of-bag error: each configuration requires a single training and the OOB is an honest estimate (each example is predicted only by the trees that have not seen it). I look at the curve as a function of n_estimators for stability, then search max_features and min_samples_leaf on a small grid. I use cross-validation or a test set only for the final confirmation and when the rows are not independent.

</details>

<details><summary><b>B2</b> · What are the most important hyperparameters of a random forest?</summary>

max_features (it controls the correlation between trees and the bias of each) and min_samples_leaf (it regularizes the leaves), then n_estimators as a budget to choose high enough by looking at the OOB error. The maximum depth is rarely needed. For regression the default of max_features in scikit-learn is 1.0 (all the features: bagging), so it is almost always worth trying smaller values like 0.33.

</details>

<details><summary><b>B3</b> · How do you choose the threshold of a random forest on an imbalanced problem?</summary>

I use the out-of-bag probabilities of the training set (oob_decision_function_), which are out-of-sample predictions: I choose the threshold on them based on costs or F1 and then validate it once on the test set. I avoid rebalancing at the start: the forest already ranks well, and the weights distort the probabilities. If I need calibrated probabilities I recalibrate them (sigmoid or isotonic) with data not used for training.

</details>

<details><summary><b>C1</b> · When does the out-of-bag error mislead?</summary>

When the examples are not independent: duplicates, repeated measurements of the same subject, consecutive frames, temporal data. The copy of an "out-of-sample" example for a tree is often in its bootstrap, so the tree has already seen it and the OOB error is too low. In those cases I validate by group (GroupKFold on customer/patient) or by time, and I remove the exact and near duplicates.

</details>

<details><summary><b>C2</b> · How do you interpret importances with highly correlated variables?</summary>

With caution: the importance (by permutation or from impurity) is split among redundant variables and each looks unimportant. I group the correlated features (correlations, clustering) and compute the group permutation, shuffling all the columns of the group together, or I reduce the redundancy upstream. I do not remove features based on the individual importance without this check.

</details>

<details><summary><b>D1</b> · How do you decide how many trees to use in a forest?</summary>

I look at the OOB error curve as n_estimators grows (with warm_start it is a single incremental training) and pick a number after which it is flat. If I need stability of the predictions (rankings, reproducible decisions) I also check the disagreement between two forests with different seeds, which settles later than the mean error. The number is a budget: more trees do not overfit but cost time, memory and latency.

</details>

<details><summary><b>D2</b> · How do you choose which features to remove from a model?</summary>

Not from the importance alone: impurity is biased toward features with many values, permutation and drop-column underestimate groups of correlated features. I group the redundant variables, remove one group (or one feature) at a time retraining and checking the performance on validation data, and keep the smallest set that does not get worse. The final cut also depends on the cost of collecting and maintaining each feature.

</details>

<details><summary><b>E1</b> · You compared two random forests and one does better by 0.5%. Do you trust it?</summary>

First I measure the variability due to the seed: I repeat with many seeds (same seed for both, paired comparison) and with a high number of trees, and I report the mean difference with the standard error. If the difference is smaller than or of the same order as the noise, I do not consider it a result; if it is distinguishable but small, I weigh it against the cost (time, complexity).

</details>

<details><summary><b>E2</b> · What distinguishes a random forest from a bagging of trees in scikit-learn?</summary>

Only the random subsampling of the features at each split (`max_features`). In the classifier the default is sqrt(p), so RandomForestClassifier really is a random forest; in the regressor the default is 1.0 (all the features), so without changing it it is equivalent to a bagging of trees. For regression I try values like 0.33 or 0.5 and choose them with the OOB or cross-validation.

</details>

<details><summary><b>E3</b> · Why must the permutation importance be computed on a validation set?</summary>

Because it measures how much the performance gets worse by shuffling a column: on the training set a flexible model has almost perfect performance and also uses noise columns to memorize, so their permutation makes it worse and they look important. On unseen data the performance reflects only what the model learned that generalizes. A practical test is to add random columns and check that they come out at the bottom of the ranking.

</details>

<details><summary><b>F1</b> · What does the random forest add to bagging, and Extra Trees to the random forest?</summary>

Bagging reduces the variance by averaging trees on different bootstraps; the random forest adds the random subsampling of the features at each split to decorrelate the trees; Extra Trees also picks the threshold at random (it does not use the bootstrap by default): it increases the bias of each tree and reduces the variance of the average further, and it is faster. In practice the big jump is from a single tree to an ensemble; the differences between the variants are a point or less and have to be checked on your own problem.

</details>

<details><summary><b>F2</b> · In which cases would you not use a random forest?</summary>

When you need to extrapolate (trends, nonstationary series), when the relationship is linear with many features (a regularized linear model is better and stable), when you need a small, fast model (devices, latency) or simple explanations, and with very sparse high-dimensional data (text). I always compare it with a linear baseline and with a boosting, using the same validation.

</details>

<details><summary><b>R8</b> · What is the out-of-bag error and when is it not reliable?</summary>

It is the error computed, for each example, using only the trees that did not see it in their bootstrap sample: a validation estimate without a separate set. It is not reliable if the examples are not independent (duplicate rows, repeated measurements, time series), because a copy of the example can be in the sample of the tree.

</details>

## Module 9 · Gradient boosting and XGBoost

<details><summary><b>A1</b> · How does gradient boosting work with the squared loss?</summary>

It starts from a constant prediction (the mean of the target); at each round it computes the residuals of the current model, trains a small tree to predict the residuals and adds its prediction multiplied by a learning rate. With other losses the residuals become the negative gradient of the loss. The training error always drops, so the number of rounds is chosen with a validation set (early stopping).

</details>

<details><summary><b>A2</b> · Why does XGBoost also use the hessian?</summary>

Because it approximates the loss with a second-order expansion: the optimal value of a leaf is -G/(H+λ), where G is the sum of the gradients and H that of the hessians; it is the Newton step, much more efficient than a plain gradient step, and it lets you compute the gain of every split in closed form. The hessian also gives the meaning of min_child_weight: a minimum weight per leaf, where very confident examples count less.

</details>

<details><summary><b>B1</b> · How do you choose the number of trees in XGBoost?</summary>

I fix a low learning rate (0.03-0.1), allow many trees (thousands) and use early stopping on a separate validation set: the training stops when the metric does not improve for k rounds and the model uses the best one. The number of rounds is therefore a result, not a hyperparameter to search; for the final estimate I use a test set never seen, or a cross-validation with early stopping inside each fold.

</details>

<details><summary><b>B2</b> · How does XGBoost handle missing values?</summary>

At each split it learns a default direction for the NaN (left or right) choosing the one that minimizes the loss on the training data; so the missing values become a signal and no imputation or indicators are needed. If the missingness is informative it is an advantage, but with other models you get the same effect with imputation plus missingness indicators; I always compare the two routes in cross-validation.

</details>

<details><summary><b>B3</b> · How do you set up the tuning of XGBoost in practice?</summary>

I start from sensible defaults (learning rate 0.05, max_depth 4, subsample and colsample 0.8), choose the rounds with early stopping on a validation set, then run a random search (20-50 combinations) over max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda and gamma with cross-validation, and retrain with early stopping. I evaluate once on the test set. I expect a modest gain over the defaults and compare it with the uncertainty of the estimate.

</details>

<details><summary><b>C1</b> · Why is early stopping done on a probabilistic metric like the log-loss?</summary>

Because the overfitting of boosting shows up first in the probabilities (which become extreme and wrong) than in the thresholded classification: the accuracy can stay almost constant while the validation log-loss is already rising a lot. A probabilistic metric picks up the deterioration right away; if the business metric is another (AUC, costs), you monitor it together.

</details>

<details><summary><b>C2</b> · How does boosting behave with noisy labels?</summary>

Worse than the forest, if not regularized: each round corrects the errors of the previous ones, so with enough rounds it tries to explain the wrong labels too and the error on clean data climbs back. It is mitigated with early stopping, shallow trees, high min_child_weight and reg_lambda, subsample and a low learning rate; and it is worth comparing it with a forest, which is naturally robust to label noise.

</details>

<details><summary><b>D1</b> · How do you read the training curve of a boosting model?</summary>

The training log-loss always drops; the validation one drops, touches a minimum and then climbs back. The minimum is the right number of rounds, the gap between the two curves measures the overfitting, and the climb back is the cost of not stopping. Configurations are compared each at its own minimum (early stopping), not at a fixed number of rounds.

</details>

<details><summary><b>D2</b> · How do you explain the prediction of an XGBoost model?</summary>

At the global level with the permutation importance on a validation set (avoiding `weight`, which rewards continuous features), at the local level with the per-example contributions (`pred_contribs` or SHAP TreeExplainer): each feature contributes to the log-odds with a signed value, and the sum of the contributions plus the bias is the raw margin. With correlated features I group the variables before interpreting.

</details>

<details><summary><b>E1</b> · How does XGBoost choose the stopping round when `eval_set` has several sets?</summary>

It uses the last set (and, if there are several metrics, the last metric): so the validation set has to go last. If I put the training set there the stop never triggers, because its metric always improves; the sign is `best_iteration` close to `n_estimators`. I check with `evals_result()` that the minimum of the validation curve matches `best_iteration`.

</details>

<details><summary><b>E2</b> · After a fit with early stopping, does the booster contain the trees of the best round?</summary>

No, it contains all the trained trees (best round plus the patience rounds). The scikit-learn wrapper uses only the first best_iteration + 1 in predict, but the native API (Booster.predict, pred_contribs, SHAP) uses everything unless you pass iteration_range=(0, best_iteration + 1). Before exporting I compare the predictions of the booster with those of the wrapper, or I retrain with the optimal number of rounds.

</details>

<details><summary><b>E3</b> · What are monotonic constraints in XGBoost for and how do you verify them?</summary>

They force the prediction to be nondecreasing (or nonincreasing) with respect to a feature: useful when the domain requires it (credit, prices, insurance) and as a regularization. They cost little if the constraint is consistent with the data. I specify them by name (not by position) and verify them with a profile: I vary the feature keeping the others fixed and check that the differences do not change sign.

</details>

<details><summary><b>F1</b> · Why is histogram boosting much faster than the classic one?</summary>

It quantizes each feature into at most 256 bins once and then, for each tree, searches the threshold only among the bins by building histograms of gradients: the cost per tree is linear in n and independent of the number of distinct values. The exact method has to sort or scan all the values at every tree. The quality changes very little because 256 candidate thresholds are almost always enough.

</details>

<details><summary><b>F2</b> · When would you choose a linear model instead of boosting?</summary>

With a mostly linear and noisy signal, few data, the need to extrapolate outside the training range (trends, growth), or requirements of interpretability and stability. Boosting wins with interactions and nonlinearity and with enough data. In any case I always put a regularized linear baseline in the table and also evaluate the hybrid (linear plus boosting on the residuals).

</details>

<details><summary><b>R9</b> · Why do you fix a high number of rounds in boosting and use early stopping?</summary>

Because the right number of rounds depends on the data and on the learning rate, and with too many rounds the model overfits (unlike a forest). You set a high ceiling, a low learning rate and let the training stop when the validation metric stops improving, then you evaluate once on the test set.

</details>

## Module 10 · Final projects

<details><summary><b>P1</b> · How do you choose between a linear model and XGBoost when the CV results are almost equal?</summary>

I look at the difference compared with the standard deviation across the folds (or I use a repeated CV): if it is within the noise, there is no evidence that the complex one is better, so I choose the simple one for interpretability, training speed and stability. I choose the complex one only if the advantage exceeds the standard error or if there is a domain reason (known interactions, a lot of data). The decision is then evaluated with the costs, not with the metric alone.

</details>

<details><summary><b>P2</b> · How would you estimate the error of a price model to be used in a city not present in the training set?</summary>

With a geographic block validation (GroupKFold on cells or cities): I keep entire areas out and measure the error on those. The random partition is optimistic because neighbors look alike. I report the mean and the dispersion across the areas, and I check that the prediction intervals keep the declared coverage outside the area too.

</details>

<details><summary><b>P3</b> · You receive a notebook with AUC 0.97 on a problem where you expect 0.8. What do you do?</summary>

First I doubt the number: I run the permuted labels test on the whole flow (it must give 0.5), I check that the preprocessing and any rebalancing are inside the validation (after the split), I look for duplicate rows between training and validation and for features built with the target or with later information. Then I measure the effect of each suspect on its own and report the honest result, with the baseline next to it.

</details>

