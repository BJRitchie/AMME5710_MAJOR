import numpy as np

def minimal_rotation(a, b):
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    
    if np.allclose(a, b): return np.eye(3)
    if np.allclose(a, -b):
        # 180 deg: pick arbitrary orth perpendicular vector u
        u = np.array([1.,0.,0.])
        if abs(np.dot(u,a))>0.9: u = np.array([0.,1.,0.])
        u = u - np.dot(u,a)*a
        u /= np.linalg.norm(u)
        # rotation by pi about u
        return -np.eye(3) + 2*np.outer(u,u)
    
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
    K = np.array([[ 0, -v[2], v[1]],
                  [ v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    R = np.eye(3) + K + K.dot(K)/(1+c) 
    return R

a = np.array([
    35.25, 0, -27.41
])
b = np.array([
    27.41, 0, -35.25
])

rmat = minimal_rotation(a, b) 
print("Rotation matrix: ")
print(rmat) 

print( f"Test: R @ a = {rmat @ a}, b = {b}, b - R @ a = { b - rmat @ a }",  ) # This

# So R @ a = b, R is the rotation matrix from a to b. 