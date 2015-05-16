
import networkx as nx

def data2graphs(datapath, start, period, stop):
        graphs = []
        datafile = open(datapath, 'r')
        for line in datafile:
                t = line.split(';')
                if int(t[0]) >= start:
                        break
        time = start
        G = nx.Graph()
        graphs.append((G, start))
        G.add_node(int(t[1]))
        G.add_node(int(t[2]))
        G.add_edge(int(t[1]), int(t[2]))
        for line in datafile:
                t = line.split(';')
                if int(t[0]) > stop:
                        break
                elif int(t[0]) > time + period:
                        time = time + period
                        G = nx.Graph()
                        graphs.append((G, time))
                G.add_node(int(t[1]))
                G.add_node(int(t[2]))
                G.add_edge(int(t[1]), int(t[2]))
        return graphs

def kgraph(graphs, k):
        kgraphs = []
        for i in range(0, len(graphs) - (k - 1)):
                G = nx.Graph()
                for j in range(0, k):
                        G = nx.compose(G, graphs[i + j][0])
                kgraphs.append((G, graphs[i + k - 1][1]))
        return kgraphs

def count_pair_freq(graphs, core):
        freq = []
        for i in range(0, len(core)):
                for j in range(i + 1, len(core)):
                        comm = 0
                        for gc in graphs:
                                if gc[0].has_edge(core[i], core[j]):
                                        comm = comm + 1
                        if comm != 0:
                                freq.append((i, j, comm))
        return sorted(freq, key = lambda tp:tp[2], reverse = True)

def find_target_pairs(freq1, total1, freq2, total2, perc):
        dict1 = {}
        thresh = total1 * perc
        for tp in freq1:
                if tp[2] >= thresh:
                        dict1[(tp[0], tp[1])] = 1
        list2 = []
        thresh = total2 * perc
        for tp in freq2:
                if tp[2] >= thresh and dict1.has_key((tp[0], tp[1])):
                        list2.append(tp)
        return list2

def get_code_seq(graphs, user1, user2):
        code = []
        for gc in graphs:
                if gc[0].has_edge(user1, user2):
                        code.append(1)
                else:
                        code.append(0)
        return code

def account_result(code, predict):
        positive = 0
        recall = 0
        alarm = 0
        false_alarm = 0
        error_bit = 0
        total_bit = len(predict)
        for i in range(0, len(code)):
                if code[i] == 1:
                        positive = positive + 1
                        if predict[i] == 1:
                                recall = recall + 1
                if predict[i] == 1:
                        alarm = alarm + 1
                        if code[i] == 0:
                                false_alarm = false_alarm + 1
                if code[i] != predict[i]:
                        error_bit = error_bit + 1
        return [recall, positive, false_alarm, alarm, error_bit, total_bit]

import matplotlib.pyplot as plt

def plot_code_seq(code):
        plt.bar(range(0,len(code)), code)
        plt.show()
        
# import matplotlib
# matplotlib.use('PDF')
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# def plot_graphs(graphs, path):
        # pages = PdfPages(path)
        # skip = 10 
        # for gc in graphs:
                # nx.draw(gc[0])
                # plt.suptitle(str(gc[1]))
                # plt.savefig(pages, format = 'pdf')
                # plt.clf()
                # skip = skip - 1
                # if skip < 0:
                        # break
        # pages.close()



