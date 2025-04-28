import numpy as np
import scipy.sparse.linalg
import time
from scipy import sparse
from src.utils import proj_vh, mat2ten, ten2mat


def tensor_factorization(M, P, M_ten, P_ten, k, SVD_S, SVD_Vt):
    M_ten = np.array(M_ten, dtype=np.float32)
    P_ten = np.array(P_ten, dtype=np.float32)
    n, m_mat = M_ten.shape
    m_mat = int(m_mat/4)
    sse = n * m_mat * 10 ** 12
    vh_init = np.sqrt(SVD_S[:k]).reshape(k, 1) * SVD_Vt[:k, :]

    nnz = np.sum(P)
    cols = np.arange(n)
    data = np.ones(n)

    for svd_flag in range(2):
        if svd_flag:
            vh_init = - vh_init
        vh = proj_vh(vh_init)


        # Solve U and V
        ite = 0
        fun = 1e10
        fun_err = 1
        vh_err = 1
        maxit = 10 ** 4
        eps = 10 ** (-6)
        U_member = np.zeros((n, k))
        for i in range(k):
            U_member[:, i] = np.sum(((M_ten - vh[i, :]) * P_ten) ** 2, axis=1)
        min_index = np.argmin(U_member, axis=1)
        while (ite < maxit) & (vh_err > eps) & (fun_err > eps) & (fun > eps):  #
            vh_old = vh.copy()
            fun_old = fun

            xhat = vh[min_index, :]
            xfill = M_ten + xhat * (1 - P_ten)

            # update V
            fre = np.array(np.bincount(min_index, minlength=k), dtype=float)
            fre[fre != 0] = 1 / fre[fre != 0]
            onehot = sparse.coo_matrix((data, (min_index, cols)), shape=(k, n)).tocsr()
            sums = onehot @ xfill  # k x m
            vh = sums / fre[:, None]
            vh = np.asarray(vh)  # remove sparse type if needed

            U_member = np.zeros((n, k))
            for i in range(k):
                U_member[:, i] = np.sum(((M_ten - vh[i, :]) * P_ten) ** 2, axis=1)
            min_index = np.argmin(U_member, axis=1)

            xhat = vh[min_index, :]

            fun = 0.5 * np.sum(((xhat - M_ten) * P_ten) ** 2) # / nnz
            fun_err = abs(fun - fun_old) / max(1, fun_old)
            vh_err = np.linalg.norm(vh - vh_old, 'fro') / m_mat / k
            ite = ite + 1

        vh = proj_vh(vh)
        vh = ten2mat(vh)

        U_member = np.zeros((n, len(vh[:, 0])))
        for i in range(len(vh[:, 0])):
            U_member[:, i] = (M - vh[i, :] == 0).sum(axis=1)
        true_ind = np.argmax(U_member, axis=1)
        R = vh[true_ind, :]  # Completed read matrix
        P_matrix = M.copy()
        P_matrix[P_matrix != 0] = 1  # projection matrix
        tmp_cri = len(np.where((R - M) != 0)[0])

        if tmp_cri < sse:
            sse = tmp_cri
            reconV = vh
            U_member = true_ind

    return reconV, sse, U_member


def reconstruct_haplotype(SNVmatrix, return_label=False):

    (num_read, m_mat) = SNVmatrix.shape  # number of reads, length of haplotypes
    M = SNVmatrix.copy()  # initial read matrix
    P = np.double(SNVmatrix != 0)  # projection matrix
    P_tensor_unfold = np.tile(P[:, :, np.newaxis], (1, 4)).reshape(num_read, -1)
    M_tensor_unfold = np.dstack((np.double(M == 1), np.double(M == 2), np.double(M == 3), np.double(M == 4))).reshape(num_read, -1)

    k_flag = 0
    k_init = 3
    k_search_thre = 1 - 10e-3

    SVD_U, SVD_S, SVD_Vt = scipy.sparse.linalg.svds(M_tensor_unfold, k=k_init, return_singular_vectors=True)
    SVD_U = SVD_U[:, ::-1]
    SVD_S = SVD_S[::-1]
    SVD_Vt = SVD_Vt[::-1, :]
    tmp_svd_M = M_tensor_unfold - SVD_U @ (SVD_S.reshape(len(SVD_S), 1) * SVD_Vt)

    k_search = k_init
    k_table = np.array([1, num_read])  # tracking Kmin and Kmax

    s1 = time.time()
    problem_total_time = 0
    SSE = len(
        np.where((M_tensor_unfold - proj_vh(np.mean(M_tensor_unfold, axis=0).reshape((1, -1)))) * P_tensor_unfold != 0)[
            0])

    while (k_table[1] - k_table[0] > 1):
        criterion = np.zeros(2)
        for k in range(k_search, k_search + 2):
            sub_time = time.time()
            if k > len(SVD_S):
                tmp_U, tmp_S, tmp_Vt = scipy.sparse.linalg.svds(tmp_svd_M, k=k - len(SVD_S),
                                                                return_singular_vectors=True)
                SVD_U = np.concatenate((SVD_U, tmp_U[:, ::-1]), axis=1)
                SVD_S = np.concatenate((SVD_S, tmp_S[::-1]))
                SVD_Vt = np.concatenate((SVD_Vt, tmp_Vt[::-1, :]), axis=0)
                tmp_svd_M = tmp_svd_M - tmp_U @ (tmp_S.reshape(len(tmp_S), 1) * tmp_Vt)

            # factorization
            Vh, SSE, ind = tensor_factorization(M, P, M_tensor_unfold, P_tensor_unfold, k, SVD_S, SVD_Vt)
            if k == k_search:
                V1 = Vh.copy()
                ind1 = ind.copy()
                SSE_old = SSE
            print('k: ' + str(k) + '\t' + 'SSE: ' + str(SSE) + '\t'
                  + 'subproblem time: ' + str(time.time() - sub_time) + '\n')
            criterion[k - k_search] = SSE

        ## Compute SSE_ratio
        SSE_ratio = SSE/SSE_old
        if SSE_ratio > k_search_thre:
            k_table[1] = k_search
            k_search = np.floor((k_search + max(1, k_table[0])) / 2)
            k_flag = 1
            Vh_record = np.array(V1, dtype=int)
            # ind_recrd = np.array(ind1, dtype=int)
        else:
            if k_flag == 0:
                k_table[0] = k_search
                k_search = 2 * k_search
            else:
                k_table[0] = k_search
                k_search = np.floor((k_search + k_table[1]) / 2)
        k_search = int(k_search)

        print('k_search: ' + str(k_search) + '\n')

    s2 = time.time() - s1
    problem_total_time = problem_total_time + s2

    reconV = np.array(Vh_record, dtype=int)
    k_final = len(reconV)
    print('Final k is:' + str(k_final) + '\t' + 'time: ' + str(problem_total_time) + '\n')

    if return_label:
        U_member = np.zeros((num_read, k_final))
        for i in range(k_final):
            U_member[:, i] = np.sum(((M - reconV[i, :]) * P) ** 2, axis=1)
        label = np.argmin(U_member, axis=1)
        return reconV, label
    else:
        return reconV


def split_merge_refine(SNVs, num_subs=5):
    V_subs = np.empty(num_subs, dtype=object)
    U_subs = np.empty(num_subs, dtype=object)
    for ind_sub in range(num_subs):
        SNVmatrix = SNVs[ind_sub]
        V_subs[ind_sub], U_subs[ind_sub] = reconstruct_haplotype(SNVmatrix, return_label=True)
    Vs = np.concatenate(V_subs, axis=0)
    SNVmatrix = np.concatenate(SNVs, axis=0)
    P = np.double(SNVmatrix != 0)
    k = Vs.shape[0]
    n = SNVmatrix.shape[0]

    U_member = np.zeros((n, k))
    for i in range(k):
        U_member[:, i] = np.sum(((SNVmatrix - Vs[i, :]) * P) ** 2, axis=1)
    label = np.argmin(U_member, axis=1)

    # merge
    k = Vs.shape[0]
    new_V = Vs.copy()

    # Loop through each pair of strains
    for i in range(k):
        if len(np.where(label ==i)[0]):
            for j in range(i + 1, k):
                if len(np.where(label ==j)[0]):
                    R1 = SNVmatrix[label==i,:]
                    R2 = SNVmatrix[label==j,:]
                    MEC_i = (np.array(R1 != Vs[i,:]).sum())/R1.shape[0] # calculate_MEC(SNVmatrix, Vs, label, i)
                    MEC_j = (np.array(R2 != Vs[j,:]).sum())/R2.shape[0]

                    # Construct a centroid strain and compute its MEC
                    centroid = ten2mat(proj_vh(np.mean((np.concatenate([mat2ten(SNVmatrix[label ==i,:])[2], mat2ten(SNVmatrix[label ==j,:])[2]], axis=0)), axis=0).reshape(1,-1)))
                    MEC_centroid = (np.array(np.concatenate([R1, R2], axis=0) != centroid).sum())/(R1.shape[0]+R2.shape[0])

                    # If merging improves the fit, merge the strains
                    if MEC_centroid < MEC_i and MEC_centroid < MEC_j:
                        new_V = np.delete(new_V, [i, j], axis=0)  # Remove the strains
                        new_V = np.concatenate([new_V, centroid], axis=0)  # Add the merged centroid
                        break

    k = new_V.shape[0]
    U_member = np.zeros((n, k))
    for i in range(k):
        U_member[:, i] = np.sum(((SNVmatrix - Vs[i, :]) * P) ** 2, axis=1)
    label = np.argmin(U_member, axis=1)

    #refine
    while new_V.shape[0]>1:
        sse_before = np.array(SNVmatrix != new_V[label,:]).sum()

        # Identify the group with the lowest frequency (use U to determine the group frequencies)
        fre = np.array(np.bincount(label, minlength=k), dtype=float)
        group_to_remove = np.argmin(fre)  # Identify the group with the lowest frequency

        # Remove the strain corresponding to the group with the lowest frequency
        refined_V = np.delete(new_V, group_to_remove, axis=0)

        # Compute the sum of squared errors (SSE) after refinement
        k = refined_V.shape[0]
        U_member = np.zeros((n, k))
        for i in range(k):
            U_member[:, i] = np.sum(((SNVmatrix - refined_V[i, :]) * P) ** 2, axis=1)
        label = np.argmin(U_member, axis=1)
        sse_after = np.array(SNVmatrix != refined_V[label,:]).sum()

        # Only accept the refinement if the SSE improvement is acceptable
        if sse_after < sse_before:
             new_V = refined_V
        else:
            break  # Return the original tensor if no improvement

    return new_V


if __name__ == "__main__":

    np.random.seed(1)
    file = '../data/simu_5strains_read_matrix.txt'
    SNVmatrix = np.loadtxt(file)
    recon_V = reconstruct_haplotype(SNVmatrix)
    # np.savetxt('../results/recon_V.txt', recon_V, fmt='%d')

    # SNVs = np.load("../data/real_5strains_read_matrix.npy", allow_pickle=True)
    # recon_V = split_merge_refine(SNVs)
    # print(recon_V.shape)
    # # np.savetxt('../results/recon_V.txt', recon_V, fmt='%d')


