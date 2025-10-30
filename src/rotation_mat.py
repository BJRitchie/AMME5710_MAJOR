import numpy as np

"""
# Example: three corresponding points in frames A and B
A = np.array([
    [37.41, 105.0, 29.1],
    [42.11, 110.3, 31.5],
    [38.9,  102.4, 28.2]
])
B = np.array([
    [35.25, 12.43, 27.59],
    [40.12, 15.80, 30.22],
    [36.10, 11.60, 26.95]
])
"""
def get_Rt(a, b): # Written by chat, unverified 
    
    # Compute centroids
    centroid_A = np.mean(a, axis=0)
    centroid_B = np.mean(b, axis=0)

    # Remove translation
    A_centered = a - centroid_A
    B_centered = b - centroid_B

    # Compute covariance matrix
    H = A_centered.T @ B_centered

    # Compute rotation (Kabsch)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure right-handed rotation (determinant +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_B - R @ centroid_A

    print("Rotation matrix R:")
    print(R)
    print("\nTranslation vector t:")
    print(t)

    # Verify mapping
    print("\nCheck mapping:")
    print("Predicted B =")
    print((R @ a.T).T + t)
    print("\nDifference:")
    print(b - ((R @ a.T).T + t))

    
    return 


print("\n\n====== Manual ====== ") 

# Two vectors to the same point in different coordinate frames 
a = np.array([ -46.08, -12.43, 27.59 ])
b = np.array([ 105, -12.57, -30.9 ])

# Rotation about X axis
roll = np.deg2rad(180)
Rx = np.array([ [1, 0, 0], 
                [0, np.cos(roll), -np.sin(roll)], 
                [0, np.sin(roll), np.cos(roll)] ])

# Rotation about Y axis
pitch = np.deg2rad(0)    
Ry = np.array([ [np.cos(pitch), 0, np.sin(pitch)], 
                [0, 1, 0], 
                [-np.sin(pitch), 0, np.cos(pitch)] ])

# Rotation about Z axis
yaw = np.deg2rad(0)      
Rz = np.array([ [np.cos(yaw), -np.sin(yaw), 0], 
                [np.sin(yaw), np.cos(yaw), 0], 
                [0, 0, 1 ] ]) 

rmat= Rz @ Ry @ Rx 
# Translation between coordinate frames 
t = b - rmat @ a

print("Rotation matrix: ")
print(rmat) 

print("\nTranslation Vector: ")
print(t)

print( f"\nManual Test: b = {b} = R @ a + t = {(rmat @ a) + t}",  ) 
print( f"\nb - (R @ a + t) = { b - ((rmat @ a) + t) }") 

