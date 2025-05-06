from funzioni_per_articolo import *
import math
import time
import pandas as pd


# -----------------------------------------------------------------------------------------
# GENERAZIONE DEL GRAFO
# Esempio di utilizzo della funzione
numero_nodi = 15  # Numero nodi  15-35
numero_nodi_disp_desiderato = 8  # Numero nodi dispari desiderato, deve essere sempre pari 8-10
numero_max_archi_desiderato = 40  # Numero massimo di archi desiderato  40-70

# Generazione del grafo single-arcs, directed con desiderato numero di nodi dispari, se non riesco con massimo numero di archi
G, Nodi_Dispari, num_odd_nodes, matrice_adiacenza, lista_archi, imbalance_nodi_disp = generate_strongly_connected_graph(
    numero_nodi, numero_nodi_disp_desiderato, numero_max_archi_desiderato)

imbalance_nodi_disp_originale = imbalance_nodi_disp
Nodi_Dispari_originale = Nodi_Dispari

imbalance_nodi_disp_esteso = []
Nodi_Dispari_esteso = []

# -----------------------------------------------------------------------------------------
# CODIFICA NODI GRAFO PER ALGORITMO RICORSIVO
# I nodi con imbalance > +-1 vengono ripetuti tante quanto è grande il loro imbalance, inoltre vengono assegnati loro dei nuovi nodi moltiplicando *100 il numero (nome) originale

for i in range(len(Nodi_Dispari)):
    if imbalance_nodi_disp[i] == -1 or imbalance_nodi_disp[i] == 1:
        imbalance_nodi_disp_esteso.append(imbalance_nodi_disp[i])
        Nodi_Dispari_esteso.append(Nodi_Dispari[i])
    else:
        sign = 1 if imbalance_nodi_disp[i] > 0 else -1
        sublist = [sign] * abs(imbalance_nodi_disp[i])
        imbalance_nodi_disp_esteso.extend(sublist)
        sublist_nodi = []
        numero_iniziale = Nodi_Dispari[i]
        for j in range(len(sublist)):
            sublist_nodi.append(numero_iniziale)
            numero_iniziale *= 100

        Nodi_Dispari_esteso.extend(sublist_nodi)

Nodi_Dispari = Nodi_Dispari_esteso
imbalance_nodi_disp = imbalance_nodi_disp_esteso

# -----------------------------------------------------------------------------------------
# TRACCIAMENTO DEL GRAFICO
# # per posizionamenti diversi dei nodi:
# pos = nx.spring_layout(G)
# # pos = nx.kamada_kawai_layout(G)
# # pos = nx.circular_layout(G)
# nx.draw(G, pos, with_labels=True)
# # nx.draw(G, pos, with_labels=True, connectionstyle='arc3, rad = 0.1') # Per curvare gli archi e far vedere gli archi doppi, rad è la curvatura, non alzarla troppo
# nx.draw_networkx_nodes(G, pos, nodelist=Nodi_Dispari, node_color='r')
#
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.show()

# Pair = lista di tutti i possibili accoppiamenti di singole coppie

# -----------------------------------------------------------------------------------------
# GENERAZIONE ACCOPPIAMENTI COPPIE POSSIBILI
# moltiplicando i numeri dispari con imbalance > +-1 per 100 tante volte quanto è grande il loro imbalance
coppie = gen_pairs(Nodi_Dispari)

# Generazione combinazione di coppie
lunghezza_coppie = (len(coppie) + 1) // 2
accoppiamenti_coppie = get_pairs(coppie, lunghezza_coppie)
# print('pairings_sum', accoppiamenti_coppie)


accoppiamenti_coppie_possibili = []
# Ogni volta che viene trovata una singola coppia non è presente tra gli archi, --> cancella quella riga
for singolo_accoppiamento_coppie in accoppiamenti_coppie:
    count = 0
    for singola_coppia in singolo_accoppiamento_coppie:
        # a = singola_coppia
        # b = Nodi_Dispari.index(singola_coppia[0])
        # c = Nodi_Dispari.index(singola_coppia[1])
        # e = imbalance_nodi_disp[b]
        # f = imbalance_nodi_disp[c]
        # segno_x = math.copysign(1, imbalance_nodi_disp[Nodi_Dispari.index(singola_coppia[0])])   # math.copysign: Rilascia solo il segno di cosa gli dai
        # segno_y = math.copysign(1, imbalance_nodi_disp[Nodi_Dispari.index(singola_coppia[1])])
        # Questa condizione è sbagliata perché non mi devo chiedere se esistono quegli archi, ma se
        # il primo elemento di ogni coppia ha imbalance > 0 e il secondo < 0 --> quella coppia va bene
        a = math.copysign(1, imbalance_nodi_disp[Nodi_Dispari.index(singola_coppia[0])])
        b = math.copysign(1, imbalance_nodi_disp[Nodi_Dispari.index(singola_coppia[1])])
        # Se nella coppia ci sono imbalance opposti, --> aumenta il contatore e ordina la coppia in modo tale che
        # il primo elemento della coppia abbia imbalance positivo, e l'altro negativo
        # math.copysign: return the value of the first parameter and the sign of the second parameter:
        if math.copysign(1, imbalance_nodi_disp[Nodi_Dispari.index(singola_coppia[0])]) != \
                math.copysign(1, imbalance_nodi_disp[Nodi_Dispari.index(singola_coppia[1])]):
            count = count + 1
            if math.copysign(1, imbalance_nodi_disp[Nodi_Dispari.index(singola_coppia[0])]) == -1 and math.copysign(1, imbalance_nodi_disp[Nodi_Dispari.index(singola_coppia[1])]) == +1:
                singola_coppia_zero_temp = singola_coppia[1]
                singola_coppia_uno_temp = singola_coppia[0]
                singola_coppia[0] = singola_coppia_zero_temp
                singola_coppia[1] = singola_coppia_uno_temp



    if count == len(Nodi_Dispari_esteso) / 2:
        accoppiamenti_coppie_possibili.append(singolo_accoppiamento_coppie)
        # g = 1 # controllo


# Controllo
# for singolo_accoppiamento_coppie in accoppiamenti_coppie_possibili:
#     for singola_coppia in singolo_accoppiamento_coppie:
#         b = Nodi_Dispari.index(singola_coppia[0])
#         c = Nodi_Dispari.index(singola_coppia[1])
#         e = imbalance_nodi_disp[b]
#         f = imbalance_nodi_disp[c]
#         segno_x = math.copysign(1, imbalance_nodi_disp[b])   # copysign: Rilascia solo il segno di cosa gli dai
#         segno_y = math.copysign(1, imbalance_nodi_disp[c])
#         if segno_x == segno_y:
#             print("ERRORE!!!!!")

# Decodifica dei nomi dei nodi dispari in modo tale che la variabile "accoppiamenti_coppie_possibili" contenga solo i nomi originali dei nodi
for singolo_accoppiamento_coppie in accoppiamenti_coppie_possibili:
    for singola_coppia in singolo_accoppiamento_coppie:

        # Modifico primo elemento della coppia
        # a = singola_coppia[0]    # Controllo
        num = str(singola_coppia[0])  # Esempio di stringa numerica

        while num.endswith("00"):  # Continua finché ci sono "00" alla fine
            num = num[:-2]  # Rimuovi gli ultimi due caratteri

        singola_coppia[0] = int(num)  # Converti la stringa risultante in un intero

        # Modifico secondo elemento della coppia
        b = singola_coppia[1]  # Controllo
        num = str(singola_coppia[1])  # Esempio di stringa numerica

        while num.endswith("00"):  # Continua finché ci sono "00" alla fine
            num = num[:-2]  # Rimuovi gli ultimi due caratteri

        singola_coppia[1] = int(num)  # Converti la stringa risultante in un intero

        # c = 1    # Controllo

Numero_Combinazioni = len(accoppiamenti_coppie_possibili)
a = 1
# -----------------------------------------------------------------------------------------
# CALCOLO DEGLI SHORTEST PATHS CON DIJKSTRA
# Al termine dell'operazione poi vedo qual'è la minore delle mie distanze

# # Calcolo dell'indegree e outdegree di ogni nodo
# in_degree = dict(G.in_degree())
# out_degree = dict(G.out_degree())
# degrees = dict(G.degree())
#
# # Calcolo del degree di ogni nodo
# degree = {node: in_degree[node] + out_degree[node] for node in G.nodes()}
# Stampa dell'indegree, outdegree e degree di ogni nodo
# for node in G.nodes():
#     if in_degree[node] > out_degree[node]:
#         print(f"Nodo {node}: indegree={in_degree[node]}, outdegree={out_degree[node]}, degree={degree[node]} (indegree>outdegree)")
#     elif in_degree[node] < out_degree[node]:
#         print(f"Nodo {node}: indegree={in_degree[node]}, outdegree={out_degree[node]}, degree={degree[node]} (indegree<outdegree)")
#     else:
#         print(f"Nodo {node}: indegree={in_degree[node]}, outdegree={out_degree[node]}, degree={degree[node]}")
#
# print(f"\nNumero di nodi con indegree>outdegree: {sum([1 for node in G.nodes() if in_degree[node] > out_degree[node]])}")
# print(f"Numero di nodi con indegree<outdegree: {sum([1 for node in G.nodes() if in_degree[node] < out_degree[node]])}")

# Calcolo della miglior combinazione di SPs
recursive_SPs_sum, recursive_SPs_augmentation = sp_pairs_distance(G, accoppiamenti_coppie_possibili, min_sums=[])
Costo_minimo_ricorsivo = np.min(recursive_SPs_sum)
SPs_Costo_minimo_ricorsivo = recursive_SPs_augmentation[recursive_SPs_sum.index(Costo_minimo_ricorsivo)]
# print("Costo minimo Algoritmo Ricorsivo: " + str(Costo_minimo_ricorsivo))
# print("SPs migliori Algoritmo Ricorsivo: " + str(SPs_Costo_minimo_ricorsivo))


# ---------------ALGORITMO ACO------------------------------------------------------------------------
# COSTRUZIONE "MATRICE DEGLI IMBALANCE-SP TRA NODI DISPARI" (D)
# -----------------------------------------------------------------------------------------
# AZIONI PRELIMINARI
# Creazione della lista "colonna": nodi_imbalance_positivo (da dove devono partire gli SPs)
nodi_imbalance_positivo = []
imbalance_positivo = []

# Creazione della lista "riga": nodi_imbalance_negativo (dove devono arrivare gli SPs)
nodi_imbalance_negativo = []
imbalance_negativo = []

# riempimento liste nodi_imbalance
for i in range(len(Nodi_Dispari_originale)):
    if imbalance_nodi_disp_originale[i] < 0:
        nodi_imbalance_negativo.append(Nodi_Dispari_originale[i])
        imbalance_negativo.append(imbalance_nodi_disp_originale[i])
    else:
        nodi_imbalance_positivo.append(Nodi_Dispari_originale[i])
        imbalance_positivo.append(imbalance_nodi_disp_originale[i])

# -----------------------------------------------------------------------------------------
# PARAMETRI ACO
start_time = time.time()
nAnt = 10  # Numero di formiche (Dimensione della popolazione)
MaxIt = 10  # Massimo numero di iterazioni
Q = 1
alpha = 1  # Peso esponenziale del feromone (Phromone Exponential Weight)
beta = 1  # Peso esponenziale euristico (Heuristic Exponential Weight)
rho = 0.05  # Tasso di evaporazione (Evaporation Rate)

# -----------------------------------------------------------------------------------------
# COSTRUZIONE "MATRICE DEGLI SP TRA NODI DISPARI" & Eta
# Creazione di D che ha le dimensioni di [nodi_imbalance_positivo x nodi_imbalance_negativo]
D = costruzione_matrice_d(nodi_imbalance_positivo, nodi_imbalance_negativo, G)

# Eta è una matrice che per ogni accoppiamento di nodi dispari rappresentato il reciproco dello SP
eta = 1 / D

# COSTRUZIONE MATRICE DEL FEROMONE
# Feromone iniziale: 10*1/(numero nodi dispari)*media_di_tutte_le_distanze) = INTENDSITA' DELLA PISTA
tau0 = 10 * Q / (len(Nodi_Dispari) * np.matrix(D).mean())
tau = tau0 * np.ones((len(nodi_imbalance_positivo), len(nodi_imbalance_negativo)))

Cost_it = np.zeros((MaxIt, 2))  # Matrice che contiene i migliori valori di costo


# STRUTTURA GLOBALE ANT
class Ant:
    Tour: object

    def __init__(self, Tour, Cost):
        self.Tour = Tour
        self.Cost = Cost


# Formiche di ogni iterazione
Winner_Ant = Ant([], [])
Winners = Ant([], [])
BestSol = Ant([], float('inf'))

# CICLO PRINCIPALE ACO
# CICLO PRINCIPALE ACO
for it in range(MaxIt):
    ant = [Ant([], 0) for i in range(nAnt)]
    for k in range(nAnt):
        # Se scrivessi solo "imbalance_positivo_temp = imbalance_positivo"  creerei un nuovo oggetto nella memoria di Python che punta allo stesso valore di imbalance_positivo.
        # Quindi entrambe le variabili punterebbero allo stesso oggetto in memoria, e qualsiasi modifica fatta su imbalance_positivo_temp si rifletterebbe anche su imbalance_positivo.
        # Per evitare questo problema per le liste puoi aggiungere dopo: ".copy()"
        imbalance_positivo_temp = imbalance_positivo.copy()
        imbalance_negativo_temp = imbalance_negativo.copy()

        P = (tau * alpha) * (eta * beta)

        # Questa condizione si può scrivere in maniera più elegante
        for N_Couples in range(0, int(max(len(nodi_imbalance_positivo), len(nodi_imbalance_negativo))), 1):
            P = P / np.sum(P, axis=None)
            i, j = roulette_wheel_selection(P)

            ant[k].Tour.append(nodi_imbalance_positivo[i])  # Aggiorno Tour
            ant[k].Tour.append(nodi_imbalance_negativo[j])
            ant[k].Cost = ant[k].Cost + D[i][j]  # Aggiorno Costo

            # Pongo a 0 righe e colonne i & j
            if imbalance_positivo_temp[i] == 1:
                P[i] = 0  # zeroes out row i
                # imbalance_positivo_temp[i] = imbalance_positivo_temp[i] - 1
            else:
                P[i, j] = 0  # annullo solo la cella
                imbalance_positivo_temp[i] = imbalance_positivo_temp[i] - 1

            if imbalance_negativo_temp[j] == -1:
                P[:, j] = 0  # zeroes out column j
                # imbalance_negativo_temp[j] = imbalance_negativo_temp[j] + 1
            else:
                P[i, j] = 0  # annullo solo la cella
                imbalance_negativo_temp[j] = imbalance_negativo_temp[j] + 1

        # Formica 1 di iterazione 1 è la prima vincitrice
        if k == 0 and it == 0:
            Winner_Ant = ant[0]
            Winners = [ant[0]]

        # Aggiornamento formica vincitrice
        if ant[k].Cost < Winner_Ant.Cost:
            Winner_Ant = ant[k]
            Winners = [Ant(ant[k].Tour, ant[k].Cost)]

        Same_Tour_Counter = 0
        for i in range(len(Winners)):
            if ant[k].Cost == Winners[i].Cost and ant[k].Tour == Winners[i].Tour:
                Same_Tour_Counter += 1
        if Same_Tour_Counter == 0 and Winner_Ant.Cost == ant[k].Cost:
            Winners.append(ant[k])
            # a = 1
        # Winner_Ants = [Winner_Ants, ant[k]]
        #     Winner_Ant.extend(ant[k])

    # AGGIORNAMENTO FEROMONE utilizzando TUTTE le formiche
    for k in range(nAnt):
        for t in range(0, len(ant[k].Tour), 2):
            i = nodi_imbalance_positivo.index(ant[k].Tour[t])
            j = nodi_imbalance_negativo.index(ant[k].Tour[t+1])

            tau[i][j] = tau[i][j] + Q/ant[k].Cost

    # EVAPORAZIONE FEROMONE
    tau = (1-rho)*tau

    # Costo medio e minimo di ogni iterazione
    Cost = np.zeros((nAnt, 1))
    for k in range(nAnt):
        Cost[k] = ant[k].Cost

    Cost_it[it][0] = np.mean(Cost)
    Cost_it[it][1] = np.min(Cost)


# print("Formica vincitrice:")
# print("Tour: " + str(Winner_Ant.Tour))
# print("Cost: " + str(Winner_Ant.Cost))

# print('DATI FINALI:')
# print('Nodi dispari:', Nodi_Dispari)
# print('Imbalance Nodi dispari:', imbalance_nodi_disp)
# print('Numero nodi dispari:', len(Nodi_Dispari))
# print("Costo minimo Algoritmo Ricorsivo: " + str(Costo_minimo_ricorsivo))
# print("SPs migliori Algoritmo Ricorsivo: " + str(SPs_Costo_minimo_ricorsivo))
# print("Costo Formica vincitrice: " + str(Winner_Ant.Cost))
# print("Tour: " + str(Winner_Ant.Tour))
end_time = time.time()
# Calcola il tempo trascorso
execution_time = end_time - start_time

print(len(Nodi_Dispari), Costo_minimo_ricorsivo, Winner_Ant.Cost, execution_time)  # , Numero_Combinazioni

# df = pd.DataFrame(lista_archi).T
# df.to_excel(excel_writer = "C:/Users/...Path.../grafo_partenza.xlsx")
#
# df_2 = pd.DataFrame(Winner_Ant.Tour).T
# df_2.to_excel(excel_writer = "C:/Users/...Path.../winner_ant.xlsx")


