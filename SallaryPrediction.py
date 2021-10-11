import pandas as pd
df = pd.read_csv('../input/d/adamfakhrifakhruddin/latihan1/Salary.csv') #memasukkan data

df.head() #menampilkan 5 data teratas

df.info() #menampilkan info dari data yang akan menunjukkan jumlah baris, nama kolom dan tipe data

#deklarasi isi X dan y
X = df.iloc[:, :-1]
y = df.iloc[:, 1]

y #untuk mengecek isi y dilakukan pemanggilan

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0) #mulai training data

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state=0)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.svm import SVC

# membuat objek SVC dan memanggil fungsi fit untuk melatih model
clf = SVC()
clf.fit(X_train, y_train)

clf.score(X_test, y_test)

y_pred = regressor.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color="Blue") #proses visualisasi data dengan diagram
plt.plot(X_train, regressor.predict(X_train), color="Red")
plt.title("Grade Level and Salary")
plt.xlabel("")
plt.ylabel("")
plt.show()

plt.scatter(X_test, y_test, color="Green")
plt.plot(X_train, regressor.predict(X_train), color="Red")
plt.title("Grade Level and Salary")
plt.xlabel("")
plt.ylabel("")
plt.show()

salary_pred = regressor.predict([[1.8]]) #proses prediksi
print("The Result you got: ", salary_pred)
