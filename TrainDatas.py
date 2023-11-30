
import pandas as pd
from sklearn.model_selection import train_test_split

pacientes = pd.read_csv("Databc.csv")

resto, prueba, resto_clase, prueba_clase = train_test_split(
pacientes[["edad","genero","presion","colesterol","diabetico"]], 
pacientes["cardiaco"],
test_size = 0.20)       

resto   