import trimesh
import numpy as np
from scipy.linalg import polar, svd, logm, expm
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from rimd_feature import cotangent_weights



def reconstruct_mesh(V_ref, dR_logs, S_list, faces, max_iter=10, tol=1e-4):
    n = len(V_ref)
    V_new = V_ref.copy()
    R_list = [np.eye(3) for _ in range(n)]
    W = cotangent_weights(V_ref, faces)

    for it in range(max_iter):
        A = lil_matrix((3*n, 3*n))
        b = np.zeros((3*n,))
        for i in range(n):
            neighbors = W[i].nonzero()[1]
            for j in neighbors:
                w = W[i, j]
                dRij = expm(dR_logs[(i, j)])
                Si = S_list[i]
                t = R_list[i] @ dRij @ Si @ (V_ref[j] - V_ref[i])
                row_i = slice(3*i, 3*i+3)
                row_j = slice(3*j, 3*j+3)
                A[row_i, row_i] += w * np.eye(3)
                A[row_i, row_j] -= w * np.eye(3)
                b[row_i] += w * t
        V_new_flat = spsolve(A.tocsr(), b)
        V_new = V_new_flat.reshape((n, 3))
        
        # Local step: update R
        for i in range(n):
            neighbors = W[i].nonzero()[1]
            Pi = []
            Qi = []
            for j in neighbors:
                Pi.append(V_ref[j] - V_ref[i])
                Qi.append(V_new[j] - V_new[i])
            Pi = np.array(Pi).T
            Qi = np.array(Qi).T
            M = Qi @ Pi.T
            U, _, Vt = svd(M)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R[:, -1] *= -1
            R_list[i] = R

        # Check convergence
        if np.linalg.norm(V_new - V_ref) < tol:
            break

    return V_new

def compute_reconstruction_error(V_true, V_recon):
    return np.mean(np.linalg.norm(V_true - V_recon, axis=1))


def compute_energy(V_new, V_ref, R_list, dR_logs, S_list, W):
    E = 0.0
    for i in range(len(V_ref)):
        neighbors = W[i].nonzero()[1]
        for j in neighbors:
            try:
                dR = expm(dR_logs[(i, j)])
            except Exception:
                dR = np.eye(3)  # logが未定義な場合の対処（近傍が足りないなど）
            e_ref = V_ref[j] - V_ref[i]
            e_new = V_new[j] - V_new[i]
            pred = R_list[i] @ dR @ S_list[i] @ e_ref
            diff = e_new - pred
            E += W[i, j] * np.dot(diff, diff)
    return E

def reconstruct_mesh(V_ref, dR_logs, S_list, faces, max_iter=20, tol=1e-4):
    n = len(V_ref)
    V_new = V_ref.copy()
    R_list = [np.eye(3) for _ in range(n)]
    W = cotangent_weights(V_ref, faces)

    # 初期エネルギー計算
    prev_E = compute_energy(V_new, V_ref, R_list, dR_logs, S_list, W)

    for it in range(max_iter):
        A = lil_matrix((3*n, 3*n))
        b = np.zeros((3*n,))
        for i in range(n):
            neighbors = W[i].nonzero()[1]
            for j in neighbors:
                w = W[i, j]
                try:
                    dRij = expm(dR_logs[(i, j)])
                except Exception:
                    dRij = np.eye(3)
                Si = S_list[i]
                t = R_list[i] @ dRij @ Si @ (V_ref[j] - V_ref[i])
                row_i = slice(3*i, 3*i+3)
                row_j = slice(3*j, 3*j+3)
                A[row_i, row_i] += w * np.eye(3)
                A[row_i, row_j] -= w * np.eye(3)
                b[row_i] += w * t

        # 頂点位置の更新
        V_new_flat = spsolve(A.tocsr(), b)
        V_new = V_new_flat.reshape((n, 3))

        # 回転行列の更新
        for i in range(n):
            neighbors = W[i].nonzero()[1]
            Pi = []
            Qi = []
            for j in neighbors:
                Pi.append(V_ref[j] - V_ref[i])
                Qi.append(V_new[j] - V_new[i])
            if len(Pi) == 0:
                continue
            Pi = np.array(Pi).T
            Qi = np.array(Qi).T
            M = Qi @ Pi.T
            U, _, Vt = svd(M)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            R_list[i] = R

        # エネルギー差による収束判定
        curr_E = compute_energy(V_new, V_ref, R_list, dR_logs, S_list, W)
        print(f"Iteration {it+1}: Energy = {curr_E:.6f}, ΔE = {abs(curr_E - prev_E):.6e}")
        if abs(curr_E - prev_E) < tol:
            break
        prev_E = curr_E

    return V_new
