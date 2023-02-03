#Tan Rémi
'''exécuter le fichier comme un script'''
#il faut glisser les deux fichiers minimnist au même endroit que le logiciel et le script (par ex:pyzo)
#Variable globale: format_reseau (format du reseau donnant le nombre), reseau ( la structure du reseau de neurones)

##IMPORTATION :
import numpy as np
import matplotlib.pyplot as plt


Donnees_train = np.genfromtxt("./mnist_minitrain_np.csv",delimiter=",")
Donnees_test = np.genfromtxt("./mnist_minitest_np.csv",delimiter=",")

np.random.shuffle(Donnees_train) #on mélange la base de données pour ne pas retomber sur le même exemple lors de l'entraînement plus tard
Images_train = Donnees_train[:,1:]/255 #pour obtenir des valeurs comprises entre 0 et 1 ( 255 car on a une valeur entre 0 et 255 pour un pixel)


Chiffres_train = Donnees_train[:,0]
Images_test = Donnees_test[:,1:]/255
Chiffres_test = Donnees_test[:,0]


## Fonctions nécessaires au réseau


def creer_tableau_de_poids(nb_neurone_depart,nb_neurone_arrivee): #créer tableau contenant tous les poids qui font les liens entre les neurones d'arrivés et les neurones de départs
    P = np.random.random((nb_neurone_arrivee,nb_neurone_depart))/100 #la division par 100 permet d'ajuster nos résultats et de ne pas obtenir des nombres trop grands à la sortie de la fonction sigmoïde et permet d'accélerer l'apprentissage du système
    return P


def derivee_sigmoide(E): # E est une matrice que l'on donne à la fonction.
    return np.exp(-E)/((1+np.exp(-E))**2)


def squish(E): # E est une matrice que l'on donne à la fonction
    return 1/(1+ np.exp(-E)) #la fonction squish permet de lisser et d'obtenir un nombre entre 0 et 1


## Initialisation du reseau

#L=[784,10] # 2 couches composées de 784 neurones puis 10 neurones

def reseau_de_neurones(format_reseau): #format_reseau est une liste contenant des nombres donnant alors le nombre de neurones par couche.
    reseau = []
    for k in range(len(format_reseau)-1): # boucle dans l'optique de créer un réseau multicouche
        reseau.append([creer_tableau_de_poids(format_reseau[k],format_reseau[k+1]),np.random.random(format_reseau[k+1])]) # on fait une liste contenant 2 éléments : la matrice des poids et la matrice des biais
    return reseau

## utilisation du reseau
def travail_du_reseau(I,reseau): #on fait fonctionner notre réseau : il doit reconnaître un chiffre à partir d'une image.
    activation = [I]              #I = image que l'on veut que le réseau reconnaisse
    for k in range(len(reseau[0])-1):
        I = squish(np.dot(reseau[k][0],I)+reseau[k][1])
        activation.append(I)
    return activation
## Vérifications

def bonne_rep(position_bonne_rep): # créer matrice idéale contenant la bonne réponse associée au chiffre elle-même associé à son image.
    C = np.zeros(10)
    C[position_bonne_rep] = 1
    return C


def verification(Position_Image,Chiffre_obtenu,Liste_chiffres): #Fonction qui permet vérifier si le chiffre obtenu est le même que celui associé à l'image renvoie soit True ou False.
    return Liste_chiffres[Position_Image] == Chiffre_obtenu


## Calcul du gradient fonction coût;

# L'activation d'un neurone sur la derniere couche dépend de différents paramètres, tels que les poids, les biais et l'activations des neurones précédents
# Dans cette fonction, nous calculons les différentes composants influençant l'activation de la dernière couche
# A l'aide de la règle de la chaine, on peut calculer l'influence du biais et du poids sur l'activation d'un neurone de la dernière couche.


def correction(taux_apprentissage,bonnes_reps_attendues,activations,N,format_reseau):
    for l in range(len(format_reseau)-1):
        derCparP = np.zeros((format_reseau[l+1],format_reseau[l]))
        derCparB = np.zeros((1,format_reseau[l+1]))
        for j in range(N):                                        #mini batchs de N
            #activation a(l-1):
            derCparActivationprecedent = np.array([])
            for k in range(format_reseau[l+1]):
                derCparActivationprecedent = np.concatenate((derCparActivationprecedent,activations[j][0]))
            derCparActivationprecedent = np.reshape(derCparActivationprecedent,(format_reseau[l+1],format_reseau[l]))


            #activation (al-y)
            correction_bis = np.zeros((1,format_reseau[l+1]))
            for k in range(format_reseau[l]):
                correction_bis = np.concatenate((correction_bis,activations[j][1]-bonnes_reps_attendues[j]))
            correction_bis = np.reshape(correction_bis[1:, : ],(format_reseau[l],format_reseau[l+1]))
            correction_bis = correction_bis.transpose()

            #sigma prime : colonne des sigma'(somme des aj.w_i, +bi)
            sigma_prime = np.zeros((1,format_reseau[l+1]))
            A = np.reshape(derivee_sigmoide(np.dot(reseau[0][0],activations[j][0])+reseau[0][1]),(1,format_reseau[l+1]))
            #On crée une matrice nulle (1,10) sinon on peut pas concaténer
            for k in range(format_reseau[l]):
                sigma_prime = np.concatenate((sigma_prime,A))
            sigma_prime = np.reshape(sigma_prime[1:, :],(format_reseau[l],format_reseau[l+1]))
            sigma_prime = sigma_prime.transpose()

            #Correction des poids
            derCparP += derCparActivationprecedent*sigma_prime*correction_bis
            #Correction des biais
            derCparB += (activations[j][1]-bonnes_reps_attendues[j])*derivee_sigmoide(np.dot(reseau[0][0],activations[j][0])+reseau[0][1])

        derCparB = derCparB/N
        derCparP = derCparP/N
        #poids
        reseau[0][0] = reseau[0][0]-taux_apprentissage*derCparP
        #biais
        reseau[0][1] = reseau[0][1]-taux_apprentissage*derCparB



## Fonction apprentissage


def apprentissage(taux_apprentissage,N,Liste_entrainement,Chiffres_entrainement,format_reseau): #N taille minibatch
#C'est la fonction qui permet au reseau de neurones de trouver les meilleurs valeurs de poids et de biais pour reconnaître un chiffre lors des entraînements.
    for k in range(int(Liste_entrainement.shape[0]/N)):
        bonnes_reps_attendues = []
        activations = []
        #boucle pour entrainement
        for i in range(k*N,(k+1)*N):
            bonnes_reps_attendues.append(np.reshape(bonne_rep(int(Chiffres_entrainement[i])),(1,10)))
            activations.append(travail_du_reseau(Liste_entrainement[i],reseau))
        correction(taux_apprentissage,bonnes_reps_attendues,activations,N,format_reseau)
        C=0
        #pour tester le taux de réussite après un entrainement
        for q in range(1000): #1000 exemples de test dans le fichier test
            I = Images_test[q]
            maxi = np.argmax(travail_du_reseau(I,reseau)[-1])
            if verification(q,maxi,Chiffres_test) == True:
                C += 1
        print(C/10,"%","nb_minibatch:",k+1)

##TEST IMAGE


def afficher (image_de_chiffre) : # image_de_chiffre = un tableau de chiffres représentant les pixels
    plt.imshow(image_de_chiffre.reshape(28,28), cmap = "gray")
    plt.show()


#Il est important d'écrire exactement <"train"> ou <"test"> pour le paramètre "Quelle_liste" pour se positionner sur le fichier souhaité


def test_image(position, Quelle_liste): #position = indice de l'image dans la liste que l'on veut tester
                                        # Cette fonction permet de tester le reseau sur un exemple pris dans l'un des deux fichiers mnsit
    plt.close()
    if Quelle_liste == "train":
       Liste_chiffres, Liste_images = Chiffres_train , Images_train
    elif Quelle_liste == "test":
        Liste_chiffres, Liste_images = Chiffres_test , Images_test
    else:
        return "Cette liste n'existe pas"
    I = Liste_images[position]
    activation = travail_du_reseau(I,reseau)
    maxi = np.argmax(activation[-1])
    afficher(I)
    print("réponse_du_reseau:",maxi)
    print(verification(position,maxi,Liste_chiffres))

##
def test() :
    global reseau
    global format_reseau
    format_reseau=[784,10] # 2 couches composées de 784 neurones puis 10 neurones
    reseau = reseau_de_neurones(format_reseau) #simplification de la notation
    apprentissage(0.75,20,Images_train,Chiffres_train,format_reseau)




























