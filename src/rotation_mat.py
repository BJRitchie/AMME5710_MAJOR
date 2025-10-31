import numpy as np

def rotation_mat_degs(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    Rx = np.array([ [1, 0, 0], 
                    [0, np.cos(roll), -np.sin(roll)], 
                    [0, np.sin(roll), np.cos(roll)] ])

    # Rotation about Y axis
    pitch = np.deg2rad(72)    
    Ry = np.array([ [np.cos(pitch), 0, np.sin(pitch)], 
                    [0, 1, 0], 
                    [-np.sin(pitch), 0, np.cos(pitch)] ])

    # Rotation about Z axis
    yaw = np.deg2rad(132)      
    Rz = np.array([ [np.cos(yaw), -np.sin(yaw), 0], 
                    [np.sin(yaw), np.cos(yaw), 0], 
                    [0, 0, 1 ] ]) 

    return Rz @ Ry @ Rx 

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

def apply_transform( s, R, t, vec, rot_mats ):
    assert len(vec.shape) == 2, "Vec must be 2D, shape 3xN. "
    assert vec.shape[0] == 3, "Vec must be of shape 3xN. " 

    # Number of points 
    N = vec.shape[1] 

    # Data storage 
    translations = np.empty_like(vec)
    rotations = np.empty_like(rot_mats)

    # Apply transform 
    for i in range(N):
        
        # Camera positions 
        if type(vec) != type(None):
            cam_pnt = vec[:, i]
            translations[:, i] = s * (R @ cam_pnt) + t.T
        
        # Camera rotations 
        if type(rot_mats) != type(None):
            cam_rot = rot_mats[:, :, i]
            rotations[:, :, i] = R @ cam_rot 
        
    return translations, rotations


if __name__ == "__main__":
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
    
    