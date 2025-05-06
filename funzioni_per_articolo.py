import random
import networkx as nx
import matplotlib.pyplot as plt  # Serve a rappresentare i grafi
import numpy as np
import scipy as sp
from random import randint, uniform
from datetime import datetime


def generate_strongly_connected_graph(num_nodi, num_nodi_disp_desiderato, num_max_archi_desiderato):
    global odd_nodes, G
    imbalance_complessivo = False
    c = 1

    while not imbalance_complessivo:  # Fino a quando non ottengo un grafo con imbalance nodi disp e pari = 0

        # PROPRIETA' STRONGLY CONNECTION-------------------------------------------------------------
        # Condizione di controllo
        if num_nodi_disp_desiderato > num_nodi:
            raise ValueError("Il numero di nodi dispari non può essere maggiore del numero totale di nodi")

        # Creazione di un percorso diretto dal primo nodo a tutti gli altri nodi
        G = nx.DiGraph()
        G.add_nodes_from(list(range(1, num_nodi + 1)))
        for i in range(1, num_nodi):
            G.add_edge(i, i + 1, weight=random.randint(1, 1000))

        G.add_edge(num_nodi, 1, weight=random.randint(1, 1000))

        # -----------------------------------------------------------------------------------------
        # AGGIUNGO ARCHI FINO A RAGGIUNDERE NUMERO DESIDERATO DI NODI DISPARI O DI ARCHI MASSIMO

        # Calcolo del numero di archi
        num_archi = G.number_of_edges()

        # Calcolo del numero di nodi dispari
        num_nodi_dispari = len([n for n in G.nodes if G.degree(n) % 2 != 0])

        while num_nodi_dispari < num_nodi_disp_desiderato and num_archi < num_max_archi_desiderato:  # Fino a quando entrambe le cond. sono vere, vai avanti, appena una viene meno esci dal ciclo
            i, j = random.sample(list(range(1, num_nodi + 1)),
                                 2)  # range(1,3) dà 1, 2      random.sample(x, 2) genera una coppia di numeri casuali
            if i != j and not G.has_edge(i, j) and not G.has_edge(j, i):  # No archi-ciclo o archi doppi
                G.add_edge(i, j, weight=random.randint(1, 1000))
                num_archi = G.number_of_edges()
                odd_nodes = []
                # num_nodi_dispari = len([n for n in G.nodes if G.degree(n) % 2 != 0])
                for n in range(1, num_nodi + 1):
                    a = G.degree(n)
                    if G.degree(n) % 2 != 0:
                        odd_nodes.append(n)

                num_nodi_dispari = len(odd_nodes)

        # -----------------------------------------------------------------------------------------
        # TRASFORMAZIONE DEL GRAFO IN UN ARRAY NUMPY
        adj_matrix = nx.to_numpy_array(G, weight='weight')
        rows, cols = np.where(adj_matrix > 0)
        edge_list = np.array(list(zip(rows + 1, cols + 1, adj_matrix[rows, cols])))

        # -----------------------------------------------------------------------------------------
        # FUNZIONE PER OTTENERE I NODI DISPARI
        # Finding odd degree vertices in graph
        # def get_odd(graph):
        degrees = [0 for i in range(len(adj_matrix))]
        out_degrees = [0 for i in range(len(adj_matrix))]
        in_degrees = [0 for i in range(len(adj_matrix))]
        # degree_imbalance = [0 for i in range(len(adj_matrix))]

        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if adj_matrix[i][j] != 0:
                    degrees[i] += 1
                    degrees[j] += 1
                    out_degrees[i] -= 1  # Per convenzione gli out_degree sono negativi
                    in_degrees[j] += 1  # Per convenzione gli in_degree sono positivi

        # in_degree - out_degree per ogni nodo
        degree_imbalance = list(np.add(np.array(in_degrees), np.array(out_degrees)))
        # odd_nodes = [i+1 for i, x in enumerate(degrees) if x % 2 != 0]

        # print("\n\ndegree dei nodi:", degrees)
        degree_dispari = [degrees[x - 1] for x in odd_nodes if x <= len(degrees)]
        degree_pari = [degrees[x - 1] for x in range(1, len(degrees) + 1) if x not in odd_nodes]
        # print("Degree dei nodi dispari:", degree_dispari)
        # print("Degree dei nodi pari:", degree_pari)

        # print("\n\nOutDegree dei nodi:", out_degrees)
        out_degree_dispari = [out_degrees[x - 1] for x in odd_nodes if x <= len(out_degrees)]
        out_degree_pari = [out_degrees[x - 1] for x in range(1, len(out_degrees) + 1) if x not in odd_nodes]
        # print("OutDegree dei nodi dispari:", out_degree_dispari)
        # print("OutDegree dei nodi pari:", out_degree_pari)

        # print("\n\nInDegree dei nodi:", in_degrees)
        in_degree_dispari = [in_degrees[x - 1] for x in odd_nodes if x <= len(in_degrees)]
        in_degree_pari = [in_degrees[x - 1] for x in range(1, len(in_degrees) + 1) if x not in odd_nodes]
        # print("InDegree dei nodi dispari:", in_degree_dispari)
        # print("InDegree dei nodi pari:", in_degree_pari)

        # print("\n\nImbalance dei nodi:", degree_imbalance)
        degree_imbalance_dispari = [degree_imbalance[x - 1] for x in odd_nodes if x <= len(degree_imbalance)]
        degree_imbalance_pari = [degree_imbalance[x - 1] for x in range(1, len(degree_imbalance) + 1) if
                                 x not in odd_nodes]
        # print("Imbalance dei nodi dispari:", degree_imbalance_dispari)
        # print("Imbalance dei nodi pari:", degree_imbalance_pari)

        imbalance_complessivo_disp = sum(degree_imbalance_dispari)
        imbalance_complessivo_pari = sum(degree_imbalance_pari)

        # print("Imbalance dei nodi dispari complessivo:", imbalance_complessivo_disp)
        # print("Imbalance dei nodi pari complessivo:", imbalance_complessivo_pari)

        if imbalance_complessivo_disp == 0 and imbalance_complessivo_pari == 0:
            imbalance_complessivo = True

        if c == 50:
            print("Il Grafo NON possiede un imbalance complessivo = 0")
            break

        c += 1

    return G, odd_nodes, len(odd_nodes), adj_matrix, edge_list, degree_imbalance_dispari


# POSIZIONAMENTI NODI DISPARI
# Prende i vertici dispari dall'elenco delle probabilità e mette tutte le combinazioni di coppie in colonna
def gen_pairs(odds):
    pairs = []
    for i in range(len(odds) - 1):
        pairs.append([])
        for j in range(i + 1, len(odds)):
            pairs[i].append([odds[i], odds[j]])

    # print('pairs are:', pairs)
    # print('\n')
    return pairs


# GENERAZIONE COMBINAZIONE DI COPPIE DI NODI DISPARI
# Prende in ingresso le la lista di combinazioni in colonna
# e con un algoritmo ricorsivo trova tutte le possibili combinazioni
def get_pairs(pairs, len_pairs, done=[], final=[], pairings_sum=[]):
    if pairs[0][0][0] not in done:
        done.append(pairs[0][0][0])

        for i in pairs[0]:
            f = final[:]
            val = done[:]
            if i[1] not in val:
                f.append(i)
            else:
                continue

            if len(f) == len_pairs:
                pairings_sum.append(f)
                return
            else:
                val.append(i[1])
                get_pairs(pairs[1:], len_pairs, val, f)

    else:
        get_pairs(pairs[1:], len_pairs, done, final)

    return pairings_sum


# SOMMA DISTANZE COMBINAZIONI SPs
def sp_pairs_distance(graph, pairings_sum, min_sums=[], aug_SPs=[]):
    for i in pairings_sum:
        s = 0
        for j in range(len(i)):
            # print('i:', i)
            # print(dijktra(graph, i[j][0], i[j][1]))
            # s += dijktra(graph, i[j][0], i[j][1])
            # Calcola il percorso più breve tra A e D
            # Calcola il percorso più breve tra A e D
            # short_p = nx.shortest_path(graph, i[j][0], i[j][1], weight='weight')

            # Calcola il peso totale dello shortest path
            s += nx.shortest_path_length(graph, i[j][0], i[j][1], weight='weight')

        min_sums.append(s)
        aug_SPs.append(i)
    return min_sums, aug_SPs


# COSTRUZIONE MATRICE D
# Creazione di D che ha le dimensioni di [nodi_imbalance_positivo x nodi_imbalance_negativo]
def costruzione_matrice_d(nodi_imb_pos, nodi_imb_neg, graph):
    distance_matrix = np.zeros((len(nodi_imb_pos), len(nodi_imb_neg)))

    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            distance_matrix[i][j] = nx.shortest_path_length(graph, nodi_imb_pos[i], nodi_imb_neg[j], weight='weight')

    return distance_matrix


# Funzione dove viene definita quale coppia di nodi dispari prendere
def roulette_wheel_selection(p):
    r = random.uniform(0, 1)
    while r == 0:
        r = random.uniform(0, 1)   # Non voglio che mi capiti la scelta Nodo Partenza = Arrivo = 0

    p_vector = p.flatten(order='C')
    c = np.cumsum(p_vector)

    # get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    # Numeri = get_indexes(r, C)
    indici_selezionati = list(np.where(c < r)[0])  # se il mio array è del tipo c = [1,2,2,2], -->
    # UTILIZZANDO <=
    # con r = 2.3, indici minori sarà [0,1,2,3,4]
    # con r = 2, indici minori sarà [0,1,2,3,4], con r = 1.5 sarà [0]
    # con r = 1, indici minori sarà [0], con r =0.7 sarà []

    # UTILIZZANDO <
    # con r = 2.3, indici minori sarà [0,1,2,3,4]
    # con r = 2, indici minori sarà [0], con r = 1.5 sarà [0]
    # con r = 1, indici minori sarà [], con r =0.7 sarà []
    # E nel mio caso è meglio questo (solo <) perchè grazie in questo modo mi prende sempre il primo elemento che eventualmente si ripete
    # if not indici_selezionati:  # Se indici minori è vuoto
    #     indice_selezionato = 0
    # else:
    #     # Se fosse un array numpy dovrei scrivere: indici_selezionati[0][-1] + 1
    #     indice_selezionato = indici_selezionati[-1] + 1

    indice_selezionato = len(indici_selezionati)  # Se fosse un array numpy dovrei scrivere: len(list(indici_selezionati[0]))

    # c_selezionati = c[indici_selezionati]

    # # Verifica se l'ultimo elemento dell'array è unico
    # if np.count_nonzero(c_selezionati == c_selezionati[-1]) == 1:
    #     # Se l'ultimo elemento è unico, restituisci l'indice dell'ultimo elemento
    #     indice = np.where(c_selezionati == c_selezionati[-1])[0][-1]
    # else:
    #     # Se l'ultimo elemento è ripetuto, restituisci l'indice del primo elemento con lo stesso valore
    #     indice = np.where(c_selezionati[:-1] == c_selezionati[-1])[0][0]
    # Quando ci si muove con le matrici si devono indicare tutti e 2 gli indici, se no ti da errore
    # j = C[0, Massimo+1]
    # Restituisce riga e colonna da rimuovere della matrice P
    # a = np.shape(p)[0]
    # b = np.shape(p)[1]
    i = int(indice_selezionato/np.shape(p)[1])  # int è una funzione che arrotonda per difetto
    if i == 0:
        j = indice_selezionato
    else:
        j = indice_selezionato % (np.shape(p)[1] * i)
    # i = 0
    # while j > len(p):
    #     j = j - len(p)
    #     i = i+1
    #
    return i, j





