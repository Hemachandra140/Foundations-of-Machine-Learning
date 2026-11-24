import pandas as pd
from sklearn.model_selection import train_test_split

D1 = pd.read_csv("Set1.csv")
D1 = D1.rename(columns={
    "Message": "email",
    "Category": "label"
})

D1["label"] = D1["label"].map({
    "ham": 0,
    "spam": 1
})

print(D1.shape)
print(D1["label"].value_counts())

D2 = pd.read_csv("Set2.csv")

word_cols = D2.columns.drop("Prediction")

D2["email"] = D2[word_cols].apply(
    lambda row: " ".join(row.index[row == 1]),
    axis=1
)

D2 = D2[["email", "Prediction"]]

D2 = D2.rename(columns={
    "email": "email",
    "Prediction": "label"
})

print(D2.shape)
print(D2["label"].value_counts())

D3 = pd.read_csv("Set3.csv")

D3 = D3.drop(columns=["Unnamed: 0"])

D3 = D3.rename(columns={
    "Body": "email",
    "Label": "label"
})


D3 = D3.dropna(subset=["email"])
D3 = D3.reset_index(drop=True)

print(D3.shape)
print(D3["label"].value_counts())

D4 = pd.read_csv("Set4.csv")

print(D4.shape)
print(D4["label"].value_counts())

D5 = pd.read_csv("Set5.csv")

D5 = D5[["text", "label_num"]].copy()

D5 = D5.rename(columns={
    "text": "email",
    "label_num": "label"
})
print(D5.shape)
print(D5["label"].value_counts())

D = pd.concat([D1, D2, D3, D4, D5], ignore_index=True)

D = D.dropna(subset=['email', 'label'])
D = D.drop_duplicates(subset=['email'])
D = D.reset_index(drop=True)

train, test = train_test_split(D,test_size=0.2,random_state=19,stratify=D['label'])

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)