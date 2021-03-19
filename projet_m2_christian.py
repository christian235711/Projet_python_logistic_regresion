import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn import datasets
from sklearn.metrics import confusion_matrix
############################################################################################################
########################################## REGRESSION LOGISTIQUE #####################################################

def sigmoide(x,theta): 
    phi =  np.insert(x ,0, np.ones(len(x[:,0])), axis=1) # on ajoute une colonne composée de 1
    z = phi.dot(theta) 
    resultat_sigmoide = np.zeros((phi.shape[0],1))
    i=0
    for a in z: # afin de ne pas générer d'erreur, on traite le sigmoide de deux manières différentes mais qui sont équivalentes mathématiquement
        if a >= 0: 
            resultat_sigmoide[i]= 1.0 / (1.0 + np.exp(-a))
        else:
            resultat_sigmoide[i]= np.exp(a) / (1 + np.exp(a))
        i=i+1    
    return resultat_sigmoide


def log_vraisemblance_v1(x, y, theta):
    return np.sum(y.reshape(N,1) * np.log(sigmoide(x, theta)) + (1-y.reshape(N,1)) * np.log(1- sigmoide(x, theta)) ) 


def log_vraisemblance_v2(x,y,theta):
    phi =  np.insert(x ,0, np.ones(len(x[:,0])), axis=1) # on ajoute une colonne composée de 1
    z = phi.dot(theta); 
    result_log= np.zeros((phi.shape[0],1))
    i=0
    for a in z:
        if(a > 500): # à partir d'une certaine valeur le sigmoide peut être approché. La valeur 500 a été un bon choix à mon sens
            result_log[i]= y[i]*a - a 
        else:
            result_log[i] = y[i]*a - np.log(1 + np.exp( a )) 
        i=i+1   

    return np.sum( result_log )

def gradient(x,y,theta): 
    phi =  np.insert(x ,0, np.ones(len(x[:,0])), axis=1) # on ajoute une colonne composée de 1
    err = y.reshape(N,1) - sigmoide(x,theta )  
    return    phi.T.dot(err)  


def hessienne(x,theta): 
    phi =  np.insert(x ,0, np.ones(len(x[:,0])), axis=1) # on ajoute une colonne composée de 1
    v= (1 - sigmoide(x,theta ))*sigmoide(x,theta ) 
    W = np.diag(v.reshape(N,))
    return -(phi.T.dot(W)).dot(phi)


def calcul_estimateur_newton(x,y,theta, iter_max, epsilon ):
    nb_iter = 0
    delta_theta = 100
    delta_gradient= 100
    delta_log_vraisemblance = 100
    while( nb_iter < iter_max and delta_theta > epsilon and delta_gradient > epsilon ):
        
        theta_ancien = theta
        DJ= gradient(x,y,theta)
        H= hessienne(x,theta)

        if(np.linalg.det(H) == 0):
            print("Attention: le déterminant de la matrice hessienne est nul, donc elle n'est pas inversible!")
            break

        theta = theta - inv(H).dot( DJ ) 
                
        delta_theta = np.linalg.norm(theta-theta_ancien)
        delta_gradient = np.linalg.norm(DJ)

        nb_iter = nb_iter + 1
    print("nombre d'itérations: ",nb_iter)
    return theta 


def calcul_estimateur_pas_fixe(x, y, theta, iter_max, pas, epsilon):
    nb_iter = 0
    delta_gradient= 100  
    
    while(nb_iter < iter_max and delta_gradient > epsilon):      
        theta_ancien = theta
        DJ= gradient(x,y,theta)
        
        theta = theta + pas* DJ  

        delta_gradient = np.linalg.norm(DJ)   
        nb_iter = nb_iter + 1; 
    print("nombre d'itérations: ", nb_iter)
    return theta


def predict_reg(x, theta):
    resultat_predict = np.ones((x.shape[0],1), dtype= int)*4
    p = sigmoide(x, theta) 
    ii = 0
    for p_element in p:
        if(p_element >0.5):
            resultat_predict[ii] = 1
        else:
            resultat_predict[ii] = 0
        ii = ii + 1        
    return  resultat_predict   

############################################################################################################
########################################## ANALYSE DISCRIMINANTE LINEAIRE #####################################################

class ADL:

    def __init__(self, x, y ):
        self.x = x
        self.y = y
        self.nb_classes = 2
        self.n = len(y)

    def pi(self):
        
        result = np.ones((self.nb_classes,1))*4
        for a in range(self.nb_classes):
            result[a] = sum( self.y==a )/self.n

        return result       


    def mu(self):
        result = np.zeros((self.nb_classes,  self.x.shape[1] ))
        for k in range(self.nb_classes):     
            compteur = 0                        
            for i in range(self.x.shape[0]):
                if self.y[i] ==k:
                    compteur = compteur + 1
                    result[k,] = result[k,] + self.x[i,]
            result[k,] = result[k,]/compteur
        return result

    def sigma(self):
        resultat = np.zeros((self.x.shape[1], self.x.shape[1] ))
        mu = self.mu()
        for k in range(self.nb_classes):
            for i in range(self.x.shape[0]):
                if( self.y[i] == k ):                    
                    diff = (self.x[i,] - mu[k]).reshape(1, self.x.shape[1] )
                    resultat= resultat + (diff ).T.dot(diff ) 
        resultat = resultat/(self.n - self.nb_classes)

        return resultat
    
    def prediction (self,z):
        resultat = np.zeros((z.shape[0] ,1), dtype = int)*4
        val = np.zeros((self.nb_classes,1))
        pi = self.pi()
        sigma= self.sigma()
        mu = self.mu()

        for i in range(z.shape[0]): # 
            for k in range(self.nb_classes): # A)La fonction linéaire discriminante qui cherche les vraisemblances
                val[k]= np.log( pi[k] )+ (z[i,].reshape(1,z.shape[1])).dot( inv(sigma).dot(mu[k].reshape(mu.shape[1],1)) ) \
                    -0.5*( mu[k].reshape(mu.shape[1],1) ).T.dot( inv(sigma).dot(mu[k].reshape(mu.shape[1],1)) )

            indice = 0
            val_max= np.amax(val)
            error = 0
            for j in range(self.nb_classes): # B) on cherche l'indice de la plus grande vraisemblance
                if( val[j] == val_max ):               
                    indice = j
                    error=error+1
            if(error > 1):
                print("Attention: l'individu {} peut être classé au moins dans deux classes ",i)        
            resultat[i]=indice
        return  resultat





############################################################################################################
#################################### APPLICATION EN UTILISANT MON CODE ###################################

# chargement de base de données iris
iris = datasets.load_iris() # iris.data: 4 colonnes. iris.target: étiquette numérotée par O, 1 ou 2. Elles sont de type ndarray.

# choix de deux variables. X est bidimensionel et y est unidimensionnel. 
X = iris.data[:, :2] # J'utilise les deux premiers colonnes.
y = (iris.target != 0) * 1 # re-étiquetage des fleurs.

# visualisation de toutes les données 
plt.figure(figsize=(10, 6));plt.title("Visualisation de toutes les données")
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='g', label='0') #
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1') #
plt.xlabel("Longueur des sépales"); plt.ylabel("Largeur des sépales"); plt.legend()

# création d'un échantionnage d'apprentissage et d'un échantionnage test

x_appren = X[10:130,] # 135 individus
y_appren = y[10:130]

x_test = np.append(X[:10,],X[130:150,], axis= 0) # 15 individus
y_test = np.append(y[:10],y[130:150] )

N = x_appren.shape[0] # nombre d'individus de notre échantionnage d'apprentissage

# visualisation des données d'apprentissage
plt.figure(figsize=(10, 6));plt.title("Visualisation des données d'apprentissage")
plt.scatter(x_appren[y_appren == 0][:, 0], x_appren[y_appren == 0][:, 1], color='g', label='0')
plt.scatter(x_appren[y_appren == 1][:, 0], x_appren[y_appren == 1][:, 1], color='r', label='1')
plt.xlim(4.1, 8.1); plt.ylim(1.86, 4.55)
plt.xlabel("Longueur des sépales"); plt.ylabel("Largeur des sépales"); plt.legend()

# visualisation des données test
plt.figure(figsize=(10, 6));plt.title("Visualisation des données test")
plt.scatter(x_test[y_test == 0][:, 0], x_test[y_test == 0][:, 1], color='g', label='0')
plt.scatter(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1], color='r', label='1')
plt.xlim(4.1, 8.1); plt.ylim(1.86, 4.55)
plt.xlabel("Longueur des sépales");plt.ylabel("Largeur des sépales");plt.legend()


########################### I. REGRESSION LOGISTIQUE ####################################

theta = np.zeros((x_appren.shape[1] + 1, 1)) #(3,1)
iter_max = 100
epsilon = 0.00001
print("\nREGRESSION LOGISTIQUE" )
print("valeur initiale de l'estimateur: ", theta.T)

theta = calcul_estimateur_newton(x_appren, y_appren, theta, iter_max, epsilon )
print("valeur finale de l'estimateur: ", theta.T)
predict_reg_log = predict_reg(x_test, theta)
print("les valeurs prédites ", predict_reg_log.T )
print("la matrice de confusion: \n ", confusion_matrix(y_test, predict_reg_log))

# visualisation de la frontière de décision  
plt.figure(figsize=(10, 6));plt.title("Frontière de décision de la regression logistique")
plt.scatter(x_appren[y_appren == 0][:, 0], x_appren[y_appren == 0][:, 1], color='g', label='0')
xx = np.linspace(4, 8 , 10)
yy =  -theta[0]/theta[2] - theta[1]/theta[2]*xx
plt.plot(xx, yy ) # droite theta[0]+theta[1]*X+theta[2]*Y=0
plt.scatter(x_appren[y_appren == 1][:, 0], x_appren[y_appren == 1][:, 1], color='r', label='1')
plt.xlim(4.1, 8.1); plt.ylim(1.86, 4.55)
plt.xlabel("Longueur des sépales"); plt.ylabel("Largeur des sépales"); plt.legend()


############################## II. ANALYSE DISCRIMINANTE LINEAIRE #################################

m1 = ADL(x_appren,y_appren)
predict_adl = m1.prediction(x_test)
print("\nANALYSE DISCRIMINANTE LINEAIRE")

print("les valeurs prédites", predict_adl.T)
print("la matrice de confusion: \n ", confusion_matrix(y_test, predict_adl ))

# visualisation de la frontière de décision
plt.figure(figsize=(10, 6)); plt.title("Frontière de décision de l'ADL")
nb =100
xxx = np.linspace(4.1, 8.1, nb)
yyy = np.linspace(1.9, 4.55, nb)
m_data = np.c_[xxx,yyy]

new_data = np.zeros((nb*nb,2 )) # on déclare une nouvelle base de données
m=0
for i in xxx:    # on remplie cette base
    for j in yyy:
        new_data[m,]= np.array([i,j])
        m=m+1
    
pp = ( m1.prediction(new_data) ).reshape(nb*nb,)

plt.scatter(new_data[pp == 0 ][:,0], new_data[pp==0][:,1], color = 'peru',marker='s' ,s=40 )
plt.scatter(new_data[pp == 1 ][:,0], new_data[pp==1][:,1], color = 'cornflowerblue', marker='s', s=40 )

plt.scatter(x_appren[y_appren == 0][:, 0], x_appren[y_appren == 0][:, 1], color='g', label='0')
plt.scatter(x_appren[y_appren == 1][:, 0], x_appren[y_appren == 1][:, 1], color='r', label='1')
plt.xlabel("Longueur des sépales");plt.ylabel("Largeur des sépales");plt.legend()
plt.show()
