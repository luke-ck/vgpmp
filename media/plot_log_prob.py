import numpy as np


# files are formatted as numpy arrays

def get_quanititties(file_path, iter):
    #  read numpy array from file
    with open(file_path + '_' + str(iter) + '.npy', 'rb') as f:
        np_arr = np.load(f)

    return np_arr


def get_log_prob_list(iterations):
    return [get_quanititties("log_prob/log_prob", i) for i in iterations]



def get_signed_distance_list(iterations):
    return [get_quanititties("signed_distance/signed_distance", i) for i in iterations]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    iterations = range(130)
    log_prob_list = get_log_prob_list(iterations)
    print(log_prob_list[0].shape)
    print(log_prob_list[0])
    signed_distance_list = get_signed_distance_list(iterations)
    # Compute frequency of log_prob for each unique signed distance
    # for each iteration

    for i, (log_prob, signed_distance) in enumerate(zip(log_prob_list, signed_distance_list)):

        # get the unique signed distances
        unique_signed_distance = np.unique(signed_distance)
        unique_log_prob = np.unique(log_prob)
        # get the frequency of each unique signed distance
        # for each iteration
        freq = np.zeros_like(unique_signed_distance)
        for j, sd in enumerate(unique_signed_distance):
            freq[j] = np.sum(signed_distance == sd)

        # get the log_prob for each unique signed distance
        log_prob_sd = np.zeros_like(unique_log_prob)
        for j, sd in enumerate(unique_log_prob):
            log_prob_sd[j] = np.sum(log_prob == sd)

        counts, bins = np.histogram(signed_distance, bins=100)
        counts = counts[1:]
        bins = bins[1:]
        plt.plot(bins[:-1], counts)
        plt.title("Signed Distance Distribution")
        plt.xlabel("Signed Distance")
        plt.ylabel("Frequency")
        plt.savefig("imgs/signed_distance/signed_distance_{}.png".format(i))
        plt.close()

        plt.plot(unique_log_prob, log_prob_sd)
        plt.title("Log Likelihood Distribution")
        plt.xlabel("Log Likelihood")
        plt.ylabel("Frequency")
        plt.savefig("imgs/log_prob/log_prob_{}.png".format(i))
        plt.close()
