import numpy as np
from BioQSA import reconstruct_haplotype


np.random.seed(1)
input_file = '../data/simu_5strains_read_matrix.txt'
SNVmatrix = np.loadtxt(input_file)
recon_V = reconstruct_haplotype(SNVmatrix)
# np.savetxt('../results/recon_V.txt', recon_V, fmt='%d')


# SNVs = np.load("../data/real_5strains_read_matrix.npy", allow_pickle=True)
# recon_V = split_merge_refine(SNVs)
# print(recon_V.shape)
# # np.savetxt('../results/recon_V.txt', recon_V, fmt='%d')
