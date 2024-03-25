import socket
import pickle
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import heapq
import random
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
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test)

def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)
def update(dW, db, W, b, learning_rate = 0.01):
    W = W - learning_rate * dW
    b = b - learning_rate * db
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
    return min(V, key=lambda v: s(v, V, f))
def multi_krum(V, f):
    n = len(V)
    s_values = [(v, s(v, V, f)) for v in V]
    min_s_values = sorted(s_values, key=lambda x: x[1])[:n - f]
    sum_vector = sum(min_v[0] for min_v in min_s_values)
    average_vector =  sum_vector / (n - f)

    return average_vector
def aggregation_function(aggregation_rule, parameters):
    if aggregation_rule == 'average':
        aggregated_model = {}
        num_clients = len(parameters)
        for key in parameters[0].keys():
            aggregated_model[key] = 0
        for client_params in parameters:
            for key, value in client_params.items():
                aggregated_model[key] += value
        for key in aggregated_model.keys():
            aggregated_model[key] /= num_clients
        return aggregated_model
    if aggregation_rule == 'krum':
        aggregated_model = {}
        num_clients = len(parameters)
        for key in parameters[0].keys():
            aggregated_model[key] = 0
            values_for_key = [d[key] for d in parameters]
            aggregated_model[key] = krum(values_for_key, 1)
        return aggregated_model
    if aggregation_rule == 'multi-krum':
        aggregated_model = {}
        num_clients = len(parameters)
        for key in parameters[0].keys():
            aggregated_model[key] = 0
            values_for_key = [d[key] for d in parameters]
            aggregated_model[key] = multi_krum(values_for_key, 1)
        return aggregated_model
    else:
        return
def main():
    server_conf = ('localhost', 12345)
    client1_conf = ('localhost', 12346)
    client2_conf = ('localhost', 12347)
    client3_conf = ('localhost', 12348)
    client4_conf = ('localhost', 12349)
    client5_conf = ('localhost', 12350)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind(server_conf)
    print('starting up on %s port %s' % server_conf)
    X_train, X_test, y_train, y_test = preprocess(df)
    Wr, br = initialisation(X_train)
    rounds = 0
    while rounds < 3:
        x=0
        parameters = []
        while x<5:
            print('\nwaiting to receive message')
            data, address = server_socket.recvfrom(8192)
        #data2, address2 = server_socket.recvfrom(8192)
            if data:
                if data[0]==0xAA:
                    print('##')
                    client1_model = pickle.loads(data[1:])
                    print('#client 1 model#')
                    parameters.append(client1_model)
                    print(client1_model)
                if data[0]==0xBA:
                    print('##')
                    client2_model = pickle.loads(data[1:])
                    print('#client 2 model#')
                    parameters.append(client2_model)
                    print(client2_model)
                if data[0]==0xCA:
                    print('##')
                    client3_model = pickle.loads(data[1:])
                    print('#client 3 model#')
                    parameters.append(client3_model)
                    print(client3_model)
                if data[0]==0xDA:
                    print('##')
                    client4_model = pickle.loads(data[1:])
                    print('#client 4 model#')
                    parameters.append(client4_model)
                    print(client4_model)
                if data[0]==0xEA:
                    print('##')
                    client5_model = pickle.loads(data[1:])
                    print('#client 5 model#')
                    parameters.append(client5_model)
                    print(client5_model)
            x=x+1
        aggregated_model = aggregation_function('multi-krum', parameters)
        print("#Global Model#")
        print(aggregated_model)
        dWr = aggregated_model['dW']
        dbr = aggregated_model['db']
        Wr, br = update(dWr, dbr, Wr, br, learning_rate = 0.01)
        server_update = {'W' : Wr, 'b' : br}
        server_update = pickle.dumps(server_update)
        server_update = b'\xcc'+server_update
        print("send global model")
        server_socket.sendto(server_update, client1_conf)
        server_socket.sendto(server_update, client2_conf)
        server_socket.sendto(server_update, client3_conf)
        server_socket.sendto(server_update, client4_conf)
        server_socket.sendto(server_update, client5_conf)
        rounds += 1
    print("FIN")
if __name__ == "__main__":
    main()
