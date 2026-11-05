import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,auc,accuracy_score,classification_report,precision_score,recall_score

df=pd.read_csv('data/merged_orders.csv')
df['order_purchase_timestamp']=pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date']=pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date']=pd.to_datetime(df['order_estimated_delivery_date'])
df['Delivery_delay']=np.where(df['order_delivered_customer_date']>df['order_estimated_delivery_date'],1,0)
df_new=df[['order_id','price','freight_value','seller_state','customer_state', 'payment_type', 'payment_installments',
       'payment_value', 'product_category_name','product_length_cm',
       'product_height_cm', 'product_width_cm','Delivery_delay']]
df_new.dropna(inplace=True)

numerical_columns = [
    'price', 'freight_value', 'payment_installments', 'payment_value',
    'product_length_cm', 'product_height_cm', 'product_width_cm'
]

categorical_columns = [
    'seller_state', 'customer_state',
    'payment_type', 'product_category_name'
]

df_encoded = pd.get_dummies(
    df_new,
    columns=categorical_columns,
    dtype=int
)
X=df_encoded.drop(columns=['order_id','Delivery_delay'])
y=df_encoded[['Delivery_delay']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

rf=RandomForestClassifier(n_estimators=100,random_state=42)

rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print(f"Accuracy is {accuracy_score(y_test,y_pred)}")
print(classification_report(y_test,y_pred))

with mlflow.start_run(run_name="delivery-delay-model"):
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mlflow.log_metrics({
        "test_accuracy": rf.score(X_test, y_test),
        "precision_score": precision_score(y_test, y_pred, zero_division=0),
        "recall_score": recall_score(y_test, y_pred, zero_division=0),
    })

    print("Tracking URI:", mlflow.get_tracking_uri())

model_dir = Path(__file__).resolve().parent.parent / "models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "random_forest.pkl"
with model_path.open("wb") as model_file:
    pickle.dump(rf, model_file)
print(f"Model saved to: {model_path}")


#Saving unseen data
X_test.to_csv('data/processed/X_test.csv',index=False)
