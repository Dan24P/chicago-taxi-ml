# Chicago Taxi Machine Learning Project

This project explores a complete supervised machine-learning workflow using publicly available Chicago taxi trip data.

The project contains two related prediction tasks:

- **Regression:** predict the total fare of a taxi trip.
- **Classification:** estimate the probability that a trip has a high fare, defined as `trip_total >= $50`.

The main goal is not only to train predictive models, but to build a reproducible workflow covering data preparation, feature engineering, model comparison, cross-validation, hyperparameter tuning, final evaluation, and model interpretation.

---

## Dataset

The project uses publicly available taxi trip records from the City of Chicago.

Each observation represents an individual taxi trip and includes information such as:

- trip duration,
- trip distance,
- total fare,
- trip start time,
- taxi company.

The raw dataset contains several million observations. Data selection, cleaning, and transformation are performed using **DuckDB** before the data is passed to the machine-learning pipeline.

Dataset source: **TODO: add Chicago Data Portal link**

---

## Data Preparation and Feature Engineering

The raw data is queried and filtered using DuckDB.

Trips with invalid or non-positive values for relevant variables such as fare, duration, and distance are removed.

The main numerical features are:

- `trip_seconds`
- `trip_miles`
- `hour_sin`
- `hour_cos`

The categorical features considered are:

- `company`
- `weekday`

### Cyclic encoding of time

Hour of day is a cyclic variable. For example, 23:00 and 00:00 are close in time even though their numerical values are far apart.

To represent this structure, hour is transformed using sine and cosine:

\[
hour_{\sin} =
\sin\left(\frac{2\pi \cdot hour}{24}\right)
\]

\[
hour_{\cos} =
\cos\left(\frac{2\pi \cdot hour}{24}\right)
\]

Categorical variables are encoded using one-hot encoding.

Preprocessing is included directly inside a scikit-learn `Pipeline` using a `ColumnTransformer`. This ensures that preprocessing is performed independently within each cross-validation fold and avoids information leakage between training and validation data.

---

## Feature Sets

Several feature configurations were compared:

- **base:** numerical features only
- **base + weekday**
- **base + company**
- **base + weekday + company**

This makes it possible to evaluate whether additional categorical information actually improves predictive performance.

The experiments showed that `company` provides useful additional information, while `weekday` contributes comparatively little.

---

# Regression Task

The regression task predicts:

```text
trip_total