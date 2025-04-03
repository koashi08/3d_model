import trimesh
import numpy as np
from scipy.linalg import polar, svd, logm, expm
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def load_mesh(filepath):
    mesh = trimesh.load_mesh(filepath, process=False)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    return vertices, faces

def cotangent_weights(vertices, faces):
    from collections import defaultdict
    n = len(vertices)
    W = lil_matrix((n, n))
    for tri in faces:
        for i in range(3):
            i0, i1, i2 = tri[i], tri[(i+1)%3], tri[(i+2)%3]
            v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
            u, v = v1 - v0, v2 - v0
            cot = np.dot(u, v) / np.linalg.norm(np.cross(u, v))
            W[i1, i2] += cot / 2
            W[i2, i1] += cot / 2
    return W

def compute_rimd_features(V_ref, V_def, faces):
    n = len(V_ref)
    W = cotangent_weights(V_ref, faces)
    R_list = []
    S_list = []
    dR_logs = {}
    for i in range(n):
        neighbors = W[i].nonzero()[1]
        if len(neighbors) < 1:
            continue
        A = []
        B = []
        for j in neighbors:
            e = (V_ref[j] - V_ref[i])
            e_ = (V_def[j] - V_def[i])
            A.append(e)
            B.append(e_)
        A = np.array(A).T
        B = np.array(B).T
        T = B @ np.linalg.pinv(A)
        R, S = polar(T)
        R_list.append(R)
        S_list.append(S)
        for j in neighbors:
            dR = R.T @ R_list[j] if j < len(R_list) else np.eye(3)
            dR_logs[(i, j)] = logm(dR)
    return dR_logs, S_list
