#@title Le modèle pour les chiffres et son entrainement

        
# On utilisera Keras
import keras
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout


# Création du modèle
model2 = Sequential()
# input: images de 128x36x1
# On applique des conv2D
model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 36, 1)))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(10, activation='softmax'))
# Enregistrer le modèle
model2.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])



#Séparer nos données entre "train" et "test"
X_train, Y_train, X_test, Y_test=(gerer_data(0.5,0,1))

print(np.asarray(Y_train).shape)

#Entrainement du modèle
model2.fit(np.asarray(X_train), np.asarray(Y_train), epochs=100, batch_size=10)


##########################################
##########################################
##########################################

#@title générer X et Y des chiffres, faire prediction, adapter resultats, afficher matrice de confusion sklearn.metrics

#Calculer les resultats (Y_prediction) avec notre modèle entrainé
Y_pred = model2.predict(np.asarray(X_test))

#Transformer les résultats : [0.001, 0.001, 0.9999,...] -> [0, 0, 1,...]
Y_pred = remettre_0_ou_1_dans_Y_pred(Y_pred)

#Transformer [ [0,1,0,0] , [0,0,1,0]]... ] en [1,2,3,1,0...]
Y_test_mat = adapter(Y_test)
Y_pred_mat = adapter(Y_pred)


#Afficher matrice
from sklearn.metrics import confusion_matrix
Matrice = confusion_matrix(Y_test_mat, Y_pred_mat)

print(Matrice)

##########################################
##########################################
##########################################

#@title Afficher matrice chiffres avec la librairie

plot_confusion_matrix_from_data(Y_test_mat, Y_pred_mat, columns=["0","1","2","3","4","5","6","7","8","9"], annot=True, cmap="Blue",
      fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='lin')
