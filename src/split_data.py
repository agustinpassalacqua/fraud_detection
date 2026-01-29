from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 2026
TEST_SIZE = 0.20          # .2 para la validacion final
VAL_SIZE_OF_REMAIN = 0.20 # del .8 restante, 20% es test para explorar hiperparams

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "creditcard.csv"
    out_dir = project_root / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    
    df["Class"] = df["Class"].astype(int)

    
    X = df.drop(columns=["Class"])
    y = df["Class"]

    #separar set de validacion
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # 2) separar train y test
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_SIZE_OF_REMAIN,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )

    # guardar
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df   = pd.concat([X_val, y_val], axis=1)
    test_df  = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    #reporte para verificar ratios de fraude similares
    def fraud_rate(s): 
        return float(s.mean())

    print("Split sizes:")
    print("  train:", len(train_df), "fraud_rate:", fraud_rate(y_train))
    print("  val:  ", len(val_df),   "fraud_rate:", fraud_rate(y_val))
    print("  test: ", len(test_df),  "fraud_rate:", fraud_rate(y_test))

if __name__ == "__main__":
    main()
