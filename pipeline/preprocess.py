# import pandas as pd


# df=pd.read_csv("student-por.csv",sep=";")
# df.to_csv("processed_student_data.csv",index=False)

# print("Data Preprocessing completed")

import pandas as pd

df=pd.read_csv("student-por.csv")

print("Original columns:")
print(df.columns.tolist())

df.to_csv("processed_student_data.csv",index=False)

print("Data Preprocessing completed")
print("Saved: processed_student_data.csv")