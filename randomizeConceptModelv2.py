# Description: This script generates random concept dictionaries for a given number of iterations and concepts.

import pandas as pd
import numpy as np
import argparse

def generate_concept_dicts(iterations, concept_num):
    app_name = "HIS17-random"
    path_results = './results/' + app_name + '/'

    # Load the dataset
    df = pd.read_csv(path_results + 'feat_data-' + app_name + '.csv', delimiter=',', quotechar='"', encoding='latin1')
    df = df.drop(['Unnamed: 0', 'Total Costs'], axis=1)  # Drop specified columns

    cols = df.columns.tolist()

    # Initialize a DataFrame to store all concept_dicts
    all_concepts_df = pd.DataFrame()

    # Generate concept_dicts for each iteration
    for iteration in range(iterations):

        # randomly shuffle cols
        np.random.shuffle(cols)

        concept_lists = [[] for _ in range(concept_num)]
        for col in cols:
            concept_lists[np.random.randint(concept_num)].append(col)

        concept_dict = {'concept_' + str(i): concept_lists[i] for i in range(concept_num)}

        # Convert the current concept_dict to a DataFrame for easier handling
        concept_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in concept_dict.items()]))
        # Add iteration number as first column to distinguish between different iterations
        concept_df.insert(0, 'iteration', iteration)

        # Append the current concept_df to all_concepts_df
        all_concepts_df = pd.concat([all_concepts_df, concept_df], axis=0, ignore_index=True)

    # Save all concept_dicts to a CSV file
    all_concepts_df.to_csv(path_results + 'random_concept_assignments.csv', index=False)

    # add empty line for better readability in random_concept_assignments.csv
    with open(path_results + 'random_concept_assignments.csv', 'a') as f:
        f.write('\n')

def main(iterations, concept_num):
    generate_concept_dicts(iterations, concept_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random concept assignments.')
    parser.add_argument('iterations', type=int, help='Number of iterations to generate concept dictionaries for.')
    parser.add_argument('concept_num', type=int, help='Number of concepts to generate in each iteration.')

    args = parser.parse_args()

    main(args.iterations, args.concept_num)
