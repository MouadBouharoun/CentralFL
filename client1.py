import socket
import pickle
import random
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from scipy import stats
from scipy.stats import t
from scipy.stats import ks_2samp
from scipy.stats import anderson_ksamp
from scipy.stats import cramervonmises
from scipy.stats import cramervonmises_2samp
#from scipy.stats import watson
#-----------------Data Preprocessing-----------------------#
df = pd.read_csv("/home/application/datasets/dataset_1.csv")

def preprocess(df):
    src_ipv4_idx = {name: idx for idx, name in enumerate(sorted(df["IPV4_SRC_ADDR"].unique()))}
    dst_ipv4_idx = {name: idx for idx, name in enumerate(sorted(df["IPV4_DST_ADDR"].unique()))}
    #attack_idx = {name: idx for idx, name in enumerate(sorted(df["Attack"].unique()))}

    df["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"].apply(lambda name: src_ipv4_idx[name])
    df["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"].apply(lambda name: dst_ipv4_idx[name])
    #df["Attack"] = df["Attack"].apply(lambda name: attack_idx[name])
    df = df.drop('Attack', axis=1)

    X=df.iloc[:, :-1].values
    y=df.iloc[:, -1].values
    X = (X - X.min()) / (X.max() - X.min())
    #X = X[5:50005, :]
    #y = y[5:50005]
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test)
#-------------------Apprentissage Machine---------------------------#
def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)
def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A
def log_loss(A, y):
    epsilon = 1e-15
    return 1/len(y) * (np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon)))
def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)
def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)
def predict(X, W, b):
    A = model(X, W, b)
    # print(A)
    return A >= 0.5
def artificial_neuron(W, b, X, y, learning_rate = 0.01, n_iter = 100, batch_size=64):
    Loss = []
    m = X.shape[0]
    for i in range(n_iter):
        for j in range(0, m, batch_size):
            # Get mini-batch
            X_batch = X[j:j+batch_size]
            y_batch = y[j:j+batch_size]

            # Forward propagation
            A = model(X_batch, W, b)
            loss = log_loss(A, y_batch)

            # Backpropagation
            dW, db = gradients(A, X_batch, y_batch)

            # Update parameters
            W, b = update(dW, db, W, b, learning_rate)

        # Compute average loss for the epoch
        epoch_loss = log_loss(model(X, W, b), y)
        Loss.append(epoch_loss)

        print(f"Iteration {i+1}/{n_iter} - Loss: {epoch_loss}")
    #print(Loss)
    plt.plot(Loss)
    plt.show()
    return (W, b, dW, db)
def server_model(W, b, dW, db):
    W, b = update(dW, db, W, b, learning_rate = 0.01)
    return (W, b)
def euclidean_distance(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))
def k_neighbors(v, V, k):
    n = len(V)
    neighbors = []
    for vector in V:
        x_v = random.uniform(0, 0.0001)
        distance = euclidean_distance(v, vector) + x_v
        heapq.heappush(neighbors, (distance, vector))

    k_nearest = heapq.nsmallest(k, neighbors)
    k_nearest_vectors = [vector for _, vector in k_nearest]
    return k_nearest_vectors
def s(v, V, f):
    n = len(V)
    sum = 0
    n_f_2_neighbors = k_neighbors(v, V, n-f-2)
    for vector in n_f_2_neighbors:
        sum += euclidean_distance(v, vector)
    return sum
def krum(V, f):
    n = len(V)
    S = []
    for v in V:
       s_v = s(v, V, f)
       S += [s_v]
    return min(S)
def generate_random(i, n):
    random_number = random.random()
    random_number *= 1/n
    random_number += i/n
    return random_number
def main():
    #----client data prep----#
    X_train, X_test, y_train, y_test = preprocess(df)
    W, b = initialisation(X_train)
    valeur1 = [generate_random(7, 100) for _ in range(7)]
    somme_totale1 = sum(valeur1)
    valeurs_normalisee1 = [valeur / somme_totale1 for valeur in valeur1]
    rounds = 0
    W1, b1 = W, b
    #----client socket----#
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_conf = ('localhost', 12345)
    client_conf = ('localhost', 12346)
    client_socket.bind(client_conf)
    #----client training----#
    while rounds < 3:
        print("Round : ", rounds)
        W1, b1, dW1, db1 = artificial_neuron(W1, b1, X_train, y_train, learning_rate = 0.01, n_iter = 40)
        params = {'dW' : dW1, 'db': db1}
        params = pickle.dumps(params)
        params = b'\xaa'+params
        
        print('##')
        client_socket.sendto(params, server_conf)
        print('client params sent')
        print('receiving model params ...')
        data, address = client_socket.recvfrom(8192)
        if data:
           if data[0]==0xCC:
               print("####################")
               global_model = pickle.loads(data[1:])
               print(global_model)
        W1 = global_model['W']
        b1 = global_model['b']
        rounds += 1

        
    print('closing the socket')
    client_socket.close()

if __name__ == "__main__":
    main()

# Calcul des moyennes et écarts types
'''
mean1 = np.mean(data1)
std_dev1 = np.std(data1, ddof=1)
n1 = len(data1)

mean2 = np.mean(data2)
std_dev2 = np.std(data2, ddof=1)
n2 = len(data2)

mean3 = np.mean(data3)
std_dev3 = np.std(data3, ddof=1)
n3 = len(data3)

mean4 = np.mean(data4)
std_dev4 = np.std(data4, ddof=1)
n4 = len(data4)

mean5 = np.mean(data5)
std_dev5 = np.std(data5, ddof=1)
n5 = len(data5)

mean6 = np.mean(data6)
std_dev6 = np.std(data6, ddof=1)
n6 = len(data6)

mean7 = np.mean(data7)
std_dev7 = np.std(data7, ddof=1)
n7 = len(data7)

mean8 = np.mean(data8)
std_dev8 = np.std(data8, ddof=1)
n8 = len(data8)
# Calcul de la statistique de test t
t_statistic_12 = (mean1 - mean2) / np.sqrt((std_dev1**2 / n1) + (std_dev2**2 / n2))

# Degrés de liberté
df_12 = n1 + n2 - 2

# Calcul de la p-valeur
p_value_12 = 2 * (1 - t.cdf(abs(t_statistic_12), df_12))
# Quantile pour alpha/2 = 0.025
alpha = 0.05
quantile_12 = t.ppf(alpha/2, df_12)

print("Quantile t pour alpha/2 = 0.025 avec ", df_12, " degrés de liberté : ", quantile_12)
print("Statistique de test t :", t_statistic_12)
print("Degrés de liberté :", df_12)
print("P-valeur 1 -> 2 :", p_value_12)
# Calcul de la statistique de test t
t_statistic_36 = (mean3 - mean6) / np.sqrt((std_dev3**2 / n3) + (std_dev6**2 / n6))

# Degrés de liberté
df_36 = n3 + n6 - 2

# Calcul de la p-valeur
p_value_36 = 2 * (1 - t.cdf(abs(t_statistic_36), df_36))
# Quantile pour alpha/2 = 0.025
alpha = 0.05
quantile_36 = t.ppf(alpha/2, df_36)

print("Quantile t pour alpha/2 = 0.025 avec ", df_36, " degrés de liberté : ", quantile_36)
print("Statistique de test t :", t_statistic_36)
print("Degrés de liberté :", df_36)
print("P-valeur 3 -> 6 :", p_value_36)
# Calcul de la statistique de test t
t_statistic_48 = (mean4 - mean8) / np.sqrt((std_dev4**2 / n4) + (std_dev8**2 / n8))

# Degrés de liberté
df_48 = n4 + n8 - 2

# Calcul de la p-valeur
p_value_48 = 2 * (1 - t.cdf(abs(t_statistic_48), df_48))
# Quantile pour alpha/2 = 0.025
alpha = 0.05
quantile_48 = t.ppf(alpha/2, df_48)

print("Quantile t pour alpha/2 = 0.025 avec ", df_48, " degrés de liberté : ", quantile_48)
print("Statistique de test t :", t_statistic_48)
print("Degrés de liberté :", df_48)
print("P-valeur 4 -> 8 :", p_value_48)
# Calcul de la statistique de test t
t_statistic_56 = (mean5 - mean6) / np.sqrt((std_dev5**2 / n5) + (std_dev6**2 / n6))

# Degrés de liberté
df_56 = n5 + n6 - 2

# Calcul de la p-valeur
p_value_56 = 2 * (1 - t.cdf(abs(t_statistic_56), df_56))
# Quantile pour alpha/2 = 0.025
alpha = 0.05
quantile_56 = t.ppf(alpha/2, df_56)

print("Quantile t pour alpha/2 = 0.025 avec ", df_56, " degrés de liberté : ", quantile_56)
print("Statistique de test t :", t_statistic_56)
print("Degrés de liberté :", df_56)
print("P-valeur 5 -> 6 :", p_value_56)
'''
'''
dif_sb11_sb81 = sample_b11 - sample_b81

# Calcul de la moyenne des différences
mean_dif_sb11_sb81 = np.mean(dif_sb11_sb81)

# Calcul de l'écart-type des différences
std_dev_dif_sb11_sb81 = np.std(dif_sb11_sb81, ddof=1)  # Utilise ddof=1 pour estimer l'écart-type non biaisé

# Calcul de l'erreur standard de la différence
se_sb11_sb81 = std_dev_dif_sb11_sb81 / np.sqrt(len(sample_b11))

# Calcul de la statistique de test t
t_statistic_sb11_sb81 = mean_dif_sb11_sb81 / se_sb11_sb81

# Dégrés de liberté
df = len(sample_b11) - 1

# Quantile pour alpha/2 = 0.025
alpha = 0.05
quantile = t.ppf(alpha/2, df)

print("Quantile t pour alpha/2 = 0.025 avec ", df, " degrés de liberté : ", quantile)
print("Différence moyenne entre les échantillons:", mean_dif_sb11_sb81)
print("Erreur standard de la différence:", se_sb11_sb81)
print("Statistique de test t:", t_statistic_sb11_sb81)
'''
'''
#-----------------------------------------------------------------------
statistic_b1, p_value_b1 = ks_2samp(sample_b1, sample_b2)
statistic_b2, p_value_b2 = ks_2samp(sample_b2, sample_b3)
statistic_b3, p_value_b3 = ks_2samp(sample_b3, sample_b4)
statistic_b4, p_value_b4 = ks_2samp(sample_b4, sample_b5)
statistic_b5, p_value_b5 = ks_2samp(sample_b5, sample_b6)
statistic_b6, p_value_b6 = ks_2samp(sample_b6, sample_b7)
statistic_b7, p_value_b7 = ks_2samp(sample_b7, sample_b8)
statistic_b11, p_value_b11 = ks_2samp(sample_b11, sample_b21)
#-------------------------------------------------------------------------
statistic_W1, p_value_W1 = ks_2samp(sample_W1.flatten(), sample_W2.flatten())
statistic_W2, p_value_W2 = ks_2samp(sample_W2.flatten(), sample_W3.flatten())
statistic_W3, p_value_W3 = ks_2samp(sample_W3.flatten(), sample_W4.flatten())
statistic_W4, p_value_W4 = ks_2samp(sample_W4.flatten(), sample_W5.flatten())
statistic_W5, p_value_W5 = ks_2samp(sample_W5.flatten(), sample_W6.flatten())
statistic_W6, p_value_W6 = ks_2samp(sample_W6.flatten(), sample_W7.flatten())
statistic_W7, p_value_W7 = ks_2samp(sample_W7.flatten(), sample_W8.flatten())
statistic_W11, p_value_W11 = ks_2samp(sample_W11.flatten(), sample_W21.flatten())
#----------------------------------------------------------------------
print("# Test de Anderson-Darling")
anderson_darling_b1 = anderson_ksamp([sample_b1, sample_b2])
anderson_darling_b2 = anderson_ksamp([sample_b2, sample_b3])
anderson_darling_b3 = anderson_ksamp([sample_b3, sample_b4])
anderson_darling_b4 = anderson_ksamp([sample_b4, sample_b5])
anderson_darling_b5 = anderson_ksamp([sample_b5, sample_b6])
anderson_darling_b6 = anderson_ksamp([sample_b6, sample_b7])
anderson_darling_b7 = anderson_ksamp([sample_b7, sample_b8])
anderson_darling_b11 = anderson_ksamp([sample_b11, sample_b21])
#----------------------------------------------------------------------
anderson_darling_W1 = anderson_ksamp([sample_W1.flatten(), sample_W2.flatten()])
anderson_darling_W2 = anderson_ksamp([sample_W2.flatten(), sample_W3.flatten()])
anderson_darling_W3 = anderson_ksamp([sample_W3.flatten(), sample_W4.flatten()])
anderson_darling_W4 = anderson_ksamp([sample_W4.flatten(), sample_W5.flatten()])
anderson_darling_W5 = anderson_ksamp([sample_W5.flatten(), sample_W6.flatten()])
anderson_darling_W6 = anderson_ksamp([sample_W6.flatten(), sample_W7.flatten()])
anderson_darling_W7 = anderson_ksamp([sample_W7.flatten(), sample_W8.flatten()])
anderson_darling_W11 = anderson_ksamp([sample_W11.flatten(), sample_W21.flatten()])
#-------------------------------------------------------------------------
p_value_AD_b1 = anderson_darling_b1.significance_level
p_value_AD_b2 = anderson_darling_b2.significance_level
p_value_AD_b3 = anderson_darling_b3.significance_level
p_value_AD_b4 = anderson_darling_b4.significance_level
p_value_AD_b5 = anderson_darling_b5.significance_level
p_value_AD_b6 = anderson_darling_b6.significance_level
p_value_AD_b7 = anderson_darling_b7.significance_level
p_value_AD_b11 = anderson_darling_b11.significance_level
#----------------------------------------------------------------
p_value_AD_W1 = anderson_darling_W1.significance_level
p_value_AD_W2 = anderson_darling_W2.significance_level
p_value_AD_W3 = anderson_darling_W3.significance_level
p_value_AD_W4 = anderson_darling_W4.significance_level
p_value_AD_W5 = anderson_darling_W5.significance_level
p_value_AD_W6 = anderson_darling_W6.significance_level
p_value_AD_W7 = anderson_darling_W7.significance_level
p_value_AD_W11 = anderson_darling_W11.significance_level
#print("# Test de Cramér-von Mises")
#statistic_CvM_b, p_value_CvM_b = cramervonmises_2samp(sample_b1, sample_b2)
#statistic_CvM_W, p_value_CvM_W = cramervonmises_2samp(sample_W1.flatten(), sample_W2.flatten())
#print("# Test de Watson")
#statistic_Watson_b, p_value_Watson_b = watson(sample_b1, sample_b2)
#statistic_Watson_W, p_value_Watson_W = watson(sample_W1.flatten(), sample_W2.flatten())
'''
'''
print("# Interprétation des tests")
alpha = 0.05
if p_value_b1 > alpha:
    print("Les échantillons sample b1 et sample b2 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Kolmogorov-Smirnov")
else:
    print("Les échantillons sample b1 et sample b2 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Kolmogorov-Smirnov")
if p_value_W1 > alpha:
    print("Les échantillons sample W1 et sample W2 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Kolmogorov-Smirnov")
else:
    print("Les échantillons sample W1 et sample W2 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Kolmogorov-Smirnov")

if p_value_AD_b1 > alpha:
    print("Les échantillons sample b1 et sample b2 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Anderson-Darling")
else:
    print("Les échantillons sample b1 et sample b2 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Anderson-Darling")
if p_value_AD_W1 > alpha:
    print("Les échantillons sample W1 et sample W2 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Anderson-Darling")
else:
    print("Les échantillons sample W1 et sample W2 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Anderson-Darling")

if p_value_b2 > alpha:
    print("Les échantillons sample b2 et sample b3 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Kolmogorov-Smirnov")
else:
    print("Les échantillons sample b2 et sample b3 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Kolmogorov-Smirnov")
if p_value_W2 > alpha:
    print("Les échantillons sample W2 et sample W3 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Kolmogorov-Smirnov")
else:
    print("Les échantillons sample W2 et sample W3 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Kolmogorov-Smirnov")

if p_value_AD_b2 > alpha:
    print("Les échantillons sample b2 et sample b3 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Anderson-Darling")
else:
    print("Les échantillons sample b2 et sample b3 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Anderson-Darling")
if p_value_AD_W2 > alpha:
    print("Les échantillons sample W2 et sample W3 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Anderson-Darling")
else:
    print("Les échantillons sample W2 et sample W3 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Anderson-Darling")

if p_value_b3 > alpha:
    print("Les échantillons sample b3 et sample b4 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Kolmogorov-Smirnov")
else:
    print("Les échantillons sample b3 et sample b4 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Kolmogorov-Smirnov")
if p_value_W3 > alpha:
    print("Les échantillons sample W3 et sample W4 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Kolmogorov-Smirnov")
else:
    print("Les échantillons sample W3 et sample W4 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Kolmogorov-Smirnov")

if p_value_AD_b3 > alpha:
    print("Les échantillons sample b3 et sample b4 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Anderson-Darling")
else:
    print("Les échantillons sample b3 et sample b4 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Anderson-Darling")
if p_value_AD_W3 > alpha:
    print("Les échantillons sample W3 et sample W4 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Anderson-Darling")
else:
    print("Les échantillons sample W3 et sample W4 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Anderson-Darling")

if p_value_b11 > alpha:
    print("Les échantillons sample b11 et sample b21 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Kolmogorov-Smirnov")
else:
    print("Les échantillons sample b11 et sample b21 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Kolmogorov-Smirnov")
if p_value_W11 > alpha:
    print("Les échantillons sample W11 et sample W21 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Kolmogorov-Smirnov")
else:
    print("Les échantillons sample W11 et sample W21 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Kolmogorov-Smirnov")

if p_value_AD_b11 > alpha:
    print("Les échantillons sample b11 et sample b21 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Anderson-Darling")
else:
    print("Les échantillons sample b11 et sample b21 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Anderson-Darling")
if p_value_AD_W11 > alpha:
    print("Les échantillons sample W11 et sample W21 semblent provenir de la même distribution (hypothèse nulle non rejetée) selon le test de Anderson-Darling")
else:
    print("Les échantillons sample W11 et sample W21 ne semblent pas provenir de la même distribution (hypothèse nulle rejetée) selon le test de Anderson-Darling")


'''
