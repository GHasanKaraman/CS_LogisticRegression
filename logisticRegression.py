import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datas = pd.read_csv("datasets/marks.csv") #Verimizi okuduk

Y = datas["Durum"]  # Logic kısmını okuyoruz 1 veya 0 kısmını

Y = Y.replace("Failed",0).replace("Pass",1)# Verilerimizde Failed kısmını 0 Pass kısmını da 1 olarak değiştiriyoruz

plt.plot(datas[Y == 1]["Vize"], datas[Y == 1]["Final"], "bo") # 1 olan verileri mavi ve yuvarlak olarak çizdiriyoruz
plt.plot(datas[Y == 0]["Vize"], datas[Y == 0]["Final"], "rs") # 0 olanları kırmızı ve kare şeklinde çizdiriyoruz

X = datas.drop(["Durum"] ,axis = 1) #Logic sutununu veri setimizden atıyoruz

X = np.c_[np.ones((X.shape[0],1)),X] # bias değerimizin katsayısı 1 olduğu için her bir X satırının başına 1 değerini veriyoruz.

Y = Y[:,np.newaxis] # Logic sutununa yeni bir eksen ekle diyerek dikey veri haline getiriyoruz. Transpoz mantığı

w = np.zeros((X.shape[1],1)) #Ağırlıkları oluşturup başlangıç değerlerine 0 atıyoruz

lr = 0.00001 # learning rate değeri denenerek bulunmuştur.
epoch = 1000000 # 1000000 kez bir eğitim uygulanacak

def sigmoid(x): # sıgmoid fonksiyonu tanımlanıyor
    return 1/(1+np.exp(-x))

for i in range(epoch): # Gradient Descent algoritması uygulanıyor
    Jw = np.dot(X.T,sigmoid(np.dot(X,w))-Y)  # Cost fonksiyonumuzun w ya göre türevi alınıyor.
    
    w = w - Jw*lr # W değerleri güncelleniyor
    
q1 = [25,100] # Sınır doğrusunun X sınırları belirtiliyor
plt.plot(q1,-(w[0]+w[1]*q1)/w[2]) # Sınır doğrusu çizdiriliyor



"""

Hatamızı dalga grafiğinde daha net görmek için bir grafik daha çizdiriyoruz

"""
q2 = np.arange(0,100) 
plt.figure() # Yeni bir pencere oluşturuyoruz
plt.plot(q2,sigmoid(np.dot(X,w)))

def tahmin(x,y): 
    return sigmoid(w[0]+w[1]*x+w[2]*y)

"""
Modelimiz üzerinde artık tahmin yürütebiliriz. 
Girilen Vize ve Final notuna göre öğrencinin kalıp kalmadığı bilgisi geriye döner.

"""
