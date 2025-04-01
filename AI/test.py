import numpy as np


def find_nonzero_x(A_ub, max_attempts=20):
    if A_ub.size == 0:
        return np.ones(A_ub.shape[1]) if A_ub.shape[1] > 0 else None
    U, S, Vt = np.linalg.svd(A_ub, full_matrices=False)
    for v in Vt:
        if np.all(A_ub @ v <= 1e-8):
            return v / np.linalg.norm(v)
    for _ in range(max_attempts):
        coeff = np.random.randn(Vt.shape[0])
        x = np.dot(coeff, Vt)
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-10:
            continue
        x /= x_norm
        if np.all(A_ub @ x <= 1e-8):
            return x
    return None


def solve_case(n, m, A, B):
    C = B @ A
    for i in range(m):
        for j in range(i + 1, m):
            if np.allclose(C[i], C[j], atol=1e-8):
                A_ub = []
                for k in range(m):
                    if k != i:
                        A_ub.append(C[k] - C[i])
                A_ub.extend(-A)
                A_ub = np.array(A_ub)
                x = find_nonzero_x(A_ub)
                if x is not None:
                    Ax = A @ x
                    if np.all(Ax >= -1e-8) and not np.allclose(x, 0, atol=1e-8):
                        z = C @ x
                        max_z = np.max(z)
                        count = np.sum(np.abs(z - max_z) < 1e-8)
                        if count >= 2:
                            return x
            else:
                diff = C[i] - C[j]
                A_eq = diff.reshape(1, -1)
                A_ub = []
                for k in range(m):
                    if k != i and k != j:
                        A_ub.append(C[k] - C[i])
                A_ub.extend(-A)
                A_ub = np.array(A_ub)
                U, S, Vt = np.linalg.svd(A_eq, full_matrices=False)
                kernel = []
                tolerance = 1e-8 * max(S) if S.size > 0 else 1e-8
                for idx in range(len(S)):
                    if S[idx] <= tolerance:
                        kernel.append(Vt[idx])
                for v in kernel:
                    for sign in [1, -1]:
                        x0 = sign * v
                        x0_norm = np.linalg.norm(x0)
                        if x0_norm < 1e-10:
                            continue
                        x0 /= x0_norm
                        Ax = A @ x0
                        valid = True
                        if np.any(Ax < -1e-8):
                            scale = 1.0
                            for k in range(n):
                                if Ax[k] < -1e-8:
                                    required_scale = (-1e-8) / (Ax[k] + 1e-16)
                                    if required_scale < 0 or np.isinf(required_scale):
                                        valid = False
                                        break
                                    scale = min(scale, required_scale)
                            if not valid:
                                continue
                            x0_scaled = x0 * scale
                            Ax_scaled = A @ x0_scaled
                            if np.all(Ax_scaled >= -1e-8):
                                x0 = x0_scaled
                                Ax = Ax_scaled
                            else:
                                continue
                        if not np.all(A_ub @ x0 <= 1e-8):
                            continue
                        z = C @ x0
                        max_z = np.max(z)
                        count = np.sum(np.abs(z - max_z) < 1e-8)
                        if count >= 2 and not np.allclose(x0, 0, atol=1e-8):
                            return x0
    for _ in range(1000):
        x = np.random.randn(n)
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-10:
            continue
        x /= x_norm
        Ax = A @ x
        Ax = np.maximum(Ax, 0)
        z = B @ Ax
        max_z = np.max(z)
        count = np.sum(np.abs(z - max_z) < 1e-8)
        if count >= 2 and np.any(np.abs(x) >= 1e-8):
            return x
    return None


def main():
    np.random.seed(42)
    n, m = map(int, input().split())
    A = np.array([list(map(float, input().split())) for _ in range(n)])
    B = np.array([list(map(float, input().split())) for _ in range(m)])
    solution = solve_case(n, m, A, B)
    if solution is not None:
        if np.all(np.abs(solution) < 1e-8):
            print("NO")
        else:
            print("YES")
            print(" ".join("%.10f" % x for x in solution))
    else:
        print("NO")


if __name__ == "__main__":
    main()
