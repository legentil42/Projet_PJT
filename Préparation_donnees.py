######################################################
######################################################
######################################################
#@title fonction gerer data pour avant le modele, fonctions adapter pour conf matrice
import os
import random

def importer(nom_fich):  #entre = chemin, sortie = couple [data, [numero, speaker, occurence]]
  data = np.load("melspectrogram/" + nom_fich)
  infos = nom_fich[0:-4]
  infos = infos.split("_")
  return data, infos

        
def gerer_data(pourcent, quoi_tester, random_ou_pas=0): #quoi tester : 0 : numero, 1: speaker, 2 : occurence, pourcent c'est le pourcent de train que tu veux

  X_train, Y_train, X_test, Y_test = [],[],[],[]

  list = os.listdir('melspectrogram/')
  
  if random_ou_pas == 1:
    random.shuffle(list)
  
  for i in range(len(list)):
    X_train.append(importer(list[i])[0])
    Y_train.append(importer(list[i])[1][quoi_tester])

  if quoi_tester == 1:
    for i in range(len(Y_train)):
      if Y_train[i] == "jackson":
        Y_train[i] = np.array([1,0,0,0])
      if Y_train[i] == "nicolas":
        Y_train[i] = np.array([0,1,0,0])
      if Y_train[i] == "theo":
        Y_train[i] = np.array([0,0,1,0])
      if Y_train[i] == "yweweler":
        Y_train[i] = np.array([0,0,0,1])

  if quoi_tester == 0:
    for i in range(len(Y_train)):
      indice = int(Y_train[i])
      Y_train[i]=[0,0,0,0,0,0,0,0,0,0]
      Y_train[i][indice] = 1




  X_test = X_train[0:round((1-pourcent)*2000)]
  Y_test = Y_train[0:round((1-pourcent)*2000)]
  X_train = X_train[round((1-pourcent)*2000):]
  Y_train = Y_train[round((1-pourcent)*2000):]




  return X_train, Y_train, X_test, Y_test

def adapter(L): #transformer [0,1,0,0] ... en [1,2,3,1,0...]
  LS = []
  for i in range(len(L)):
    LS.append(list(L[i]).index(max(list(L[i]))))
  return LS

def remettre_0_ou_1_dans_Y_pred(Y_pred):

  for i in range(len(Y_pred)):
    imax = list(Y_pred[i]).index(max(list(Y_pred[i])))
    Y_pred[i][imax]= 1

    for k in range(len(Y_pred[i])):
      if Y_pred[i][k] != 1:
        Y_pred[i][k]=0

  print(Y_pred)

  tot = 0
  for i in range(len(Y_pred)):
   if list(Y_pred[i]) == list(Y_test[i]):
      tot += 1

  print(tot/len(Y_pred))
  return Y_pred


