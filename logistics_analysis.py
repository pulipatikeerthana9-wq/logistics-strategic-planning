# 10.1 LOAD AND INSPECT

import pandas as pd
import numpy as np

df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin-1"
)

print(df.shape)
print(df["Order Id"].nunique(), "distinct orders")
df.info()

for c in ["order date (DateOrders)", "shipping date (DateOrders)"]:
    df[c] = pd.to_datetime(df[c])


# 10.2 DEFINE TARGET

df["delay_days"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

df["late_flag"] = (df["delay_days"] > 0).astype(int)

print(pd.crosstab(df["late_flag"], df["Late_delivery_risk"]))
print(pd.crosstab(df["Delivery Status"], df["late_flag"]))


# 10.3 KPI SUMMARY

mode_summary = df.groupby("Shipping Mode").agg(
    order_items=("late_flag", "size"),
    late_rate=("late_flag", "mean"),
    avg_actual=("Days for shipping (real)", "mean"),
    avg_scheduled=("Days for shipment (scheduled)", "mean")
)

mode_summary["late_rate"] *= 100

print(mode_summary.sort_values("late_rate", ascending=False))


# 10.4 CLASSIFICATION - NON-CANCELLED SHIPMENTS

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("\n" + "=" * 60)
print("10.4 CLASSIFICATION - NON-CANCELLED SHIPMENTS")
print("=" * 60)

df_model = df[
    df["Delivery Status"] != "Shipping canceled"
].copy()

print("Analysis records:", len(df_model))
print("Distinct orders:", df_model["Order Id"].nunique())

num_features = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Discount",
    "Order Item Product Price"
]

cat_features = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Category Name",
    "Customer Segment",
    "Type"
]

feature_columns = num_features + cat_features

X_model = df_model[feature_columns]
y_model = df_model["late_flag"]
groups = df_model["Order Id"]

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx_model, test_idx_model = next(
    gss.split(
        X_model,
        y_model,
        groups=groups
    )
)

classification_prep = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            num_features
        ),
        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            cat_features
        )
    ]
)

classification_model = Pipeline([
    ("prep", classification_prep),
    (
        "model",
        RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )
    )
])

classification_model.fit(
    X_model.iloc[train_idx_model],
    y_model.iloc[train_idx_model]
)

class_predictions = classification_model.predict(
    X_model.iloc[test_idx_model]
)

class_probabilities = classification_model.predict_proba(
    X_model.iloc[test_idx_model]
)[:, 1]

print("\nTraining records:", len(train_idx_model))
print("Testing records:", len(test_idx_model))

print("\nClassification Report:")
print(
    classification_report(
        y_model.iloc[test_idx_model],
        class_predictions,
        target_names=["On Time", "Late"]
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_model.iloc[test_idx_model],
        class_predictions
    )
)

roc_auc = roc_auc_score(
    y_model.iloc[test_idx_model],
    class_probabilities
)

print("\nROC-AUC:", round(roc_auc, 4))


# 10.5 REGRESSION - NON-CANCELLED SHIPMENTS

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n" + "=" * 60)
print("10.5 REGRESSION - NON-CANCELLED SHIPMENTS")
print("=" * 60)

y_reg = df_model["Days for shipping (real)"]

regression_prep = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            num_features
        ),
        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            cat_features
        )
    ]
)

regression_model = Pipeline([
    ("prep", regression_prep),
    (
        "model",
        RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )
    )
])

regression_model.fit(
    X_model.iloc[train_idx_model],
    y_reg.iloc[train_idx_model]
)

reg_predictions = regression_model.predict(
    X_model.iloc[test_idx_model]
)

mae = mean_absolute_error(
    y_reg.iloc[test_idx_model],
    reg_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg.iloc[test_idx_model],
        reg_predictions
    )
)

baseline_predictions = df_model[
    "Days for shipment (scheduled)"
].iloc[test_idx_model]

baseline_mae = mean_absolute_error(
    y_reg.iloc[test_idx_model],
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_reg.iloc[test_idx_model],
        baseline_predictions
    )
)

mae_improvement = (
    (baseline_mae - mae) / baseline_mae * 100
)

rmse_improvement = (
    (baseline_rmse - rmse) / baseline_rmse * 100
)

print("\nRandom Forest Regression:")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

print("\nScheduled-Days Baseline:")
print("MAE:", round(baseline_mae, 4))
print("RMSE:", round(baseline_rmse, 4))

print(
    "\nMAE improvement over baseline:",
    round(mae_improvement, 2),
    "%"
)

print(
    "RMSE improvement over baseline:",
    round(rmse_improvement, 2),
    "%"
)


# 10.6 CLUSTERING

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("\n" + "=" * 60)
print("10.6 CLUSTERING")
print("=" * 60)

cluster_cols = [
    "Order Item Quantity",
    "Sales",
    "Order Item Product Price",
    "Order Item Discount"
]

cluster_data = df[cluster_cols].copy()

cluster_data = cluster_data.fillna(
    cluster_data.median(numeric_only=True)
)

cluster_data = cluster_data.clip(lower=0)

X_cluster = np.log1p(cluster_data)

scaler = StandardScaler()
X_cluster = scaler.fit_transform(X_cluster)

cluster_scores = {}

for k in range(2, 8):
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(X_cluster)

    score = silhouette_score(
        X_cluster,
        labels
    )

    cluster_scores[k] = score

    print(
        "k =",
        k,
        "| Silhouette Score =",
        round(score, 4)
    )

best_k = max(
    cluster_scores,
    key=cluster_scores.get
)

print(
    "\nSelected number of clusters:",
    best_k
)

print(
    "Best silhouette score:",
    round(cluster_scores[best_k], 4)
)

final_kmeans = KMeans(
    n_clusters=best_k,
    n_init=10,
    random_state=42
)

df["cluster"] = final_kmeans.fit_predict(X_cluster)

cluster_profile = df.groupby("cluster").agg(
    order_items=("Order Id", "size"),
    avg_quantity=("Order Item Quantity", "mean"),
    avg_sales=("Sales", "mean"),
    avg_product_price=("Order Item Product Price", "mean"),
    avg_discount=("Order Item Discount", "mean"),
    late_rate=("late_flag", "mean")
)

cluster_profile["late_rate"] *= 100

print("\nCluster Profile:")
print(cluster_profile.round(2))


# 10.7 ILLUSTRATIVE CAPACITY-CONSTRAINED ROUTING

from ortools.constraint_solver import (
    pywrapcp,
    routing_enums_pb2
)

print("\n" + "=" * 60)
print("10.7 ILLUSTRATIVE ROUTING")
print("=" * 60)

print("This example uses made-up distances and demands.")
print("It is NOT a routing result from the DataCo dataset.")

distance = [
    [0, 4, 6, 8],
    [4, 0, 3, 5],
    [6, 3, 0, 2],
    [8, 5, 2, 0]
]

demand = [0, 2, 3, 2]
vehicle_capacity = [4, 4]

manager = pywrapcp.RoutingIndexManager(
    len(distance),
    len(vehicle_capacity),
    0
)

routing = pywrapcp.RoutingModel(manager)


def distance_callback(i, j):
    return distance[
        manager.IndexToNode(i)
    ][
        manager.IndexToNode(j)
    ]


transit_callback_index = (
    routing.RegisterTransitCallback(
        distance_callback
    )
)

routing.SetArcCostEvaluatorOfAllVehicles(
    transit_callback_index
)


def demand_callback(i):
    return demand[
        manager.IndexToNode(i)
    ]


demand_callback_index = (
    routing.RegisterUnaryTransitCallback(
        demand_callback
    )
)

routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,
    vehicle_capacity,
    True,
    "Capacity"
)

search_parameters = (
    pywrapcp.DefaultRoutingSearchParameters()
)

search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)

solution = routing.SolveWithParameters(
    search_parameters
)

if solution:
    print("\nIllustrative Routes:")

    for vehicle in range(len(vehicle_capacity)):
        index = routing.Start(vehicle)
        route = []

        while not routing.IsEnd(index):
            route.append(
                manager.IndexToNode(index)
            )

            index = solution.Value(
                routing.NextVar(index)
            )

        route.append(0)

        print(
            "Vehicle",
            vehicle,
            ":",
            route
        )
else:
    print("No feasible routing solution found.")


# 10.8 ANALYSIS POPULATION VALIDATION

print("\n" + "=" * 60)
print("10.8 ANALYSIS POPULATION VALIDATION")
print("=" * 60)

print("\n1. Delivery Status Distribution:")
print(
    df["Delivery Status"].value_counts(
        dropna=False
    )
)

df["is_cancelled"] = (
    df["Delivery Status"] == "Shipping canceled"
)

print("\n2. Cancelled vs Non-Cancelled:")
print(
    df["is_cancelled"]
    .value_counts()
    .rename(
        index={
            False: "Non-cancelled",
            True: "Cancelled"
        }
    )
)

print("\n3. Late Flag by Cancellation Status:")
print(
    pd.crosstab(
        df["is_cancelled"],
        df["late_flag"],
        margins=True
    )
)

population_summary = (
    df.groupby("is_cancelled")
    .agg(
        rows=("Order Id", "size"),
        late_rows=("late_flag", "sum"),
        distinct_orders=("Order Id", "nunique")
    )
)

population_summary["late_rate"] = (
    population_summary["late_rows"]
    / population_summary["rows"]
    * 100
)

population_summary.index = [
    "Non-cancelled",
    "Cancelled"
]

print("\n4. Population Summary:")
print(population_summary.round(2))

print("\n5. Shipping Duration by Population:")

duration_summary = (
    df.groupby("is_cancelled")
    .agg(
        avg_actual_days=(
            "Days for shipping (real)",
            "mean"
        ),
        avg_scheduled_days=(
            "Days for shipment (scheduled)",
            "mean"
        )
    )
)

duration_summary.index = [
    "Non-cancelled",
    "Cancelled"
]

print(duration_summary.round(2))

print("\n6. Late Flag Consistency:")

consistency = pd.crosstab(
    df["Delivery Status"],
    df["late_flag"]
)

print(consistency)

print("\n" + "=" * 60)
print("ALL ANALYSIS SECTIONS COMPLETED")
print("=" * 60)