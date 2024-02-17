"""Plot data distributions for each feature"""

import os
import glob

import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    home_path = "./cev-lm"
    figure_folder = "figs"

    data_folder = "./cev-lm/data/downloaded_data"
    bins = 25

    figure_path = os.path.join(home_path, figure_folder)
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    for file_name in glob.glob(data_folder + "/*_train.tsv"):
        print(f"Processing {file_name}...")

        feature = file_name.split('/')[-1].replace("_train.tsv", "")
        with open(file_name, 'r') as f:
            values = np.array(list(map(lambda x: float(x.strip().split('\t')[-1]), f.readlines())))
            # print(np.argwhere(np.isnan(values)))
            values = values[~np.isnan(values)] # remove all nans - implicitly handled by filtration during training
            values = values[(values < 10) & (values > -10)] # remove all outliers - again handled by filtration
            print(values[:10], values.mean(), values.std(), values.max(), values.min())

            sns.histplot(values, bins=bins, log_scale=(False, True))
            plt.xlabel(f"Delta for {feature}")
            plt.ylabel("Number of samples (log-scale)")
            plt.title(f"Training samples for {feature} in log-scale")
            # plt.savefig(os.path.join(figure_path, f"{feature}.png"), bbox_inches='tight')

            plt.savefig(os.path.join(figure_path, f"{feature}.pdf"), format='pdf', bbox_inches='tight')
            plt.clf()
