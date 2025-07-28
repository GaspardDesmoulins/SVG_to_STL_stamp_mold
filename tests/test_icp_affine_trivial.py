import unittest
import numpy as np
from utils import icp_affine

class TestICPAffine(unittest.TestCase):
    def test_icp_affine_trivial(self):
        # Carré de base
        square = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        # Transformation affine connue : scale 2, translation (3, 4)
        H_true = np.array([
            [2, 0, 3],
            [0, 2, 4],
            [0, 0, 1]
        ])
        square_h = np.hstack([square, np.ones((4, 1))])
        square_transformed = (H_true @ square_h.T).T[:, :2]
        # Utilisation de icp_affine
        aligned, H_found = icp_affine(square, square_transformed)
        # Vérification
        self.assertTrue(np.allclose(H_found, H_true, atol=1e-6), f"Matrice trouvée incorrecte:\n{H_found}\nMatrice attendue:\n{H_true}")


    def test_icp_affine_rotation_30(self):
        # Carré de base
        square = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        # Matrice de rotation 30° autour de l'origine
        theta = np.deg2rad(30)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        square_h = np.hstack([square, np.ones((4, 1))])
        square_rotated = (R @ square_h.T).T[:, :2]
        # Utilisation de icp_affine
        aligned, H_found = icp_affine(square, square_rotated)
        # Vérification
        self.assertTrue(np.allclose(H_found, R, atol=1e-6), f"Matrice trouvée incorrecte (rotation 30°):\n{H_found}\nMatrice attendue:\n{R}")

    def test_icp_affine_shear_x(self):
        # Carré de base
        square = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        # Matrice de shear sur x (k=0.5)
        k = 0.5
        S = np.array([
            [1, k, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        square_h = np.hstack([square, np.ones((4, 1))])
        square_sheared = (S @ square_h.T).T[:, :2]
        # Utilisation de icp_affine
        aligned, H_found = icp_affine(square, square_sheared)
        # Vérification
        self.assertTrue(np.allclose(H_found, S, atol=1e-6), f"Matrice trouvée incorrecte (shear x):\n{H_found}\nMatrice attendue:\n{S}")
    
    def test_icp_affine_scale_y(self):
        # Carré de base
        square = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        # Matrice de scaling sur y (facteur 3)
        sy = 3.0
        S = np.array([
            [1, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ])
        square_h = np.hstack([square, np.ones((4, 1))])
        square_scaled = (S @ square_h.T).T[:, :2]
        # Utilisation de icp_affine
        aligned, H_found = icp_affine(square, square_scaled)
        # Vérification
        self.assertTrue(np.allclose(H_found, S, atol=1e-6), f"Matrice trouvée incorrecte (scale y):\n{H_found}\nMatrice attendue:\n{S}")
if __name__ == "__main__":
    unittest.main()
