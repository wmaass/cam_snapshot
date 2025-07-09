import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from scipy.spatial.distance import pdist
import numpy as np
import re
from tabulate import tabulate
#import calign21 as ca
#from compare_clusters import compare_cluster_assignments

class FeatureAnalysis:
    def __init__(self, app_name, bc):
        self.app_name = app_name
        self.path_results = f'./results/{app_name}/'
        self.data_path = os.path.join(self.path_results, 'concept_defs_last-2.csv')
        self.feature_co_occurrence = None
        self.decision_index = 0
        self.benchmark = bc

    def init_cooccurence_matrix(self):


        # Regex, um die Nummer am Ende des Dateinamens zu extrahieren
        pattern = re.compile(r"concept_defs_last-(\d+)\.csv$")
        
        att_names = []

        for file in os.listdir(self.path_results):

            match = pattern.match(file)
            if match:
                file_path = os.path.join(self.path_results, file)

                print("file_path: ", file_path)

                df = pd.read_csv(file_path, header=None)

                # drop first column of df
                df = df.drop(df.columns[0], axis=1)
                df = df.astype(str)

                # iterate df
                for index, row in df.iterrows():
                    print(row, type(row))

                    for index, value in row.items():
                        if value != 'nan' and value not in att_names:
                            att_names.append(value)

        print("len att_names: ", len(att_names))
        print("att_names: ", att_names)

        #df = pd.read_csv(self.data_path, header=None)
        #last_row = df[df.iloc[:, 0] == 'decision'].index[-1]

        # drop first column
        #df = df.drop(df.columns[0], axis=1)

        # Drucken der ersten Spalte über df.columns
        #first_column_name = df.columns[0]
        #print(df[first_column_name])

        # reindex columns of df
        #df.columns = range(len(df.columns))

        #df = df.astype(str)

        # att_names = []
        # # iterate df
        # for index, row in df.iterrows():
        #     for i in range(len(row)):
        #         if row[i] != 'nan':
        #             # add row[i] to list att_names
        #             att_names.append(row[i])

        self.feature_co_occurrence = pd.DataFrame(0, index=att_names, columns=att_names)

    def extract_and_save_concepts(self):
        # extract last concept definitions from trials
        # and save results in concept_defs_last-<trial_number>.csv

        # Regex, um die Nummer am Ende des Dateinamens zu extrahieren
        pattern = re.compile(r"concept_definitions-(\d+)\.csv$")

        # find all indexes of rows containing 'decision'
        
        for file in os.listdir(self.path_results):
            match = pattern.match(file)
            if match:
                file_path = os.path.join(self.path_results, file)
                df = pd.read_csv(file_path, header=None)

                decision_indexes = df[df[0] == 'decision'].index

                print('decision_indexes:', decision_indexes.values)

                if len(decision_indexes) == 1:
                    print('decision_indexes == 1:', decision_indexes.values)
                    # Extrahiere alle Zeilen nach der letzten 'decision' Zeile
                    last_decision_index = df[df[0] == 'decision'].index[0]
                    last_concepts_df = df.iloc[1:last_decision_index]

                else:
                    print('decision_indexes != 1:', decision_indexes.values)
                    # extract all rows between two 'decision' rows indexed by decision_indexes[self.decision_index] and decision_indexes[self.decision_index+1]

                    # Finde den Index der letzten Zeile, die 'decision' enthält
                    last_decision_index = df[df[0] == 'decision'].index[-1]
                    
                    # Extrahiere alle Zeilen nach der letzten 'decision' Zeile
                    last_concepts_df = df.iloc[last_decision_index + 1:]
                
                # Generiere den neuen Dateinamen
                new_file_name = f"concept_defs_last-{match.group(1)}.csv"
                new_file_path = os.path.join(self.path_results, new_file_name)
                
                # Speichere die extrahierten Zeilen in der neuen Datei
                last_concepts_df.to_csv(new_file_path, index=False, header=False)
                print(f"Saved extracted concepts to {new_file_path}")


    def update_cooccurence_matrix(self):
        # iterate over all files in the results folder with name 'concept_defs_last'
        # and update the co-occurrence matrix

        features = list(self.feature_co_occurrence.index)

        for file in os.listdir(self.path_results):
            if 'concept_defs_last' in file:
                print(file)
                df = pd.read_csv(os.path.join(self.path_results, file), header=None)
                # drop first column
                df = df.drop(df.columns[0], axis=1)
                # reindex columns
                df.columns = range(len(df.columns))

                #print(df.head())
                for index, row in df.iterrows():
                    for i in range(len(row)):                        
                        for j in range(i, len(row)):
                            if pd.notna(row[i]) and pd.notna(row[j]) and row[i] in features and row[j] in features:
                                self.feature_co_occurrence.at[row[i], row[j]] += 1
                                if i != j:
                                    self.feature_co_occurrence.at[row[j], row[i]] += 1



    def assign_columns_to_clusters_and_save(self, linkage_matrix, num_clusters):
        """
        Assigns columns of a given matrix to clusters based on a linkage matrix and saves the assignments to a CSV file.

        Parameters:
        - linkage_matrix: The linkage matrix obtained from hierarchical clustering.
        - matrix: The DataFrame whose columns are to be clustered.
        - num_clusters: The desired number of clusters.
        - path_results: The path where the results CSV file will be saved.
        """

        # Initialize a DataFrame from feature_co_occurrence columns
        df = pd.DataFrame(index=self.feature_co_occurrence.columns)

        # Iterate from 1 to num_clusters (inclusive) and form clusters at each iteration
        for n in range(1, num_clusters + 1):

            cluster_assignments = fcluster(linkage_matrix, t=n, criterion='maxclust')

            # Assign columns to clusters for the current number of clusters
            df[f'Cluster_{n}'] = cluster_assignments

        # read csv file bc
        bc_path = f"./results/{self.benchmark}/feat_co_occ.csv"
        #print("bc_path: ", bc_path)

        df_bc = pd.read_csv(bc_path, header=None)

        # df_bc_feats first column of df_bc without first row
        df_bc_feats = df_bc.iloc[:, 0][1:]
        
        #print("bc: ", df_bc_feats)

        # sort df by index of df_bc
        df = df.reindex(df_bc_feats)

        # if folder clusters does not exist, create it
        if not os.path.exists("./clusters"):
            os.mkdir("./clusters")
        
        # Save to CSV file
        results_file_path = f"./clusters/column_clusters_{self.app_name}.csv"
        df.to_csv(results_file_path)

        # # Assign columns to clusters
        # column_clusters = {self.feature_co_occurrence.columns[i]: cluster_assignments[i] for i in range(len(self.feature_co_occurrence.columns))}

        # # Save to CSV file
        # results_file_path = "./clusters/" + "column_clusters_" + self.app_name +".csv"
        # with open(results_file_path, 'w') as f:
        #     for key, value in column_clusters.items():
        #         f.write(f"{key},{value}\n")
        
        # print(f"Column cluster assignments saved to {results_file_path}")

    def plot_heatmap(self, clustered=False):

        if clustered:
            #print("Clustering")

            linkage_matrix = linkage(self.feature_co_occurrence, method='complete', metric='euclidean') #average euclidean

            # Plotting the dendrogram
            # plt.figure(figsize=(10, 7))
            # dendrogram(linkage_matrix)
            # plt.title('Hierarchical Clustering Dendrogram (Features)')
            # plt.xlabel('Feature Index')
            # plt.ylabel('Distance')

            # save plt to file
            # plt.savefig(os.path.join(self.path_results, 'dendrogram.png'))
            # plt.show()

            # Step 2: Decide on the number of clusters (e.g., 10 for this example)
            # This step is usually more involved and may require analyzing the dendrogram
            num_clusters = 10

            self.assign_columns_to_clusters_and_save(linkage_matrix, num_clusters)

            ordered_idx = leaves_list(linkage_matrix)
            self.feature_co_occurrence = self.feature_co_occurrence.iloc[ordered_idx, ordered_idx]

            # save the co-occurrence matrix to a file
            self.feature_co_occurrence.to_csv(os.path.join(self.path_results, 'feat_co_occ_clustered.csv'))

            file_name='heatmap_clustering.png'
            title = 'Feature Co-occurrence - Hierarch. Clustering: ' + self.app_name

            plt.figure(figsize=(12, 12))
            plt.title(title)
            plt.subplots_adjust(left=0.3, top=0.95, bottom=0.4)
            
            p = sns.heatmap(self.feature_co_occurrence, cmap='YlGnBu', linewidths=0.5, linecolor='black',
                            xticklabels=self.feature_co_occurrence.columns, yticklabels=self.feature_co_occurrence.index, annot_kws={"size": 8})

            for i in range(len(self.feature_co_occurrence)):
                plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='green', lw=0))

            plt.xticks(fontsize=8, rotation=45, ha="right")
            plt.yticks(fontsize=8)

            plt.savefig(os.path.join(self.path_results, file_name))
            plt.show()

        else:
            #print("No Clustering")
            # save the co-occurrence matrix to a file
            self.feature_co_occurrence.to_csv(os.path.join(self.path_results, 'feat_co_occ.csv'))
            #file_name='heatmap.png'
            #title = 'Feature Co-occurrence: ' + self.app_name
            
        # plt.figure(figsize=(12, 12))
        # plt.title(title)
        # plt.subplots_adjust(left=0.3, top=0.95, bottom=0.4)
        
        # p = sns.heatmap(self.feature_co_occurrence, cmap='YlGnBu', linewidths=0.5, linecolor='black',
        #                 xticklabels=self.feature_co_occurrence.columns, yticklabels=self.feature_co_occurrence.index, annot_kws={"size": 8})

        # for i in range(len(self.feature_co_occurrence)):
        #     plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='green', lw=0))

        # plt.xticks(fontsize=8, rotation=45, ha="right")
        # plt.yticks(fontsize=8)

        # plt.savefig(os.path.join(self.path_results, file_name))
        # plt.show()
        #plt.close()

def main():

    #names = ["HC-benchmark", "HIS17-onehot-two-classes"] # "HC-four-classes"] # , "HIS17-onehot-random", "HIS17-onehot"]
    #names = ["HIS17-2", "HIS17"] # ["HIS17-2"] #, "HIS17-two-classes"] # ['LOAN', 'LOAN-2'] # ["HIS17-2", "HIS17-two-classes"] # "HC-four-classes"] # , "HIS17-onehot-random", "HIS17-onehot"]
    names = ["HIS17-Benchmark", "HIS17-random"] # ["HIS17-2", "HIS17-two-classes"] # "HC-four-classes"] # , "HIS17-onehot-random", "HIS17-onehot"]

    #print("names: ", names)

    for app_name in names:
        analysis = FeatureAnalysis(app_name, 'HIS17')
        analysis.extract_and_save_concepts()
        analysis.init_cooccurence_matrix()
        analysis.update_cooccurence_matrix()
        analysis.plot_heatmap(clustered=False)  # Setzen Sie clustered auf True, um die nicht-geclusterte Heatmap zu plotten
        analysis.plot_heatmap(clustered=True)  # Setzen Sie clustered auf True, um die geclusterte Heatmap zu plotten

if __name__ == "__main__":
    main()
