import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes



def extract_svg_polygons(svg_path):
    """
    Extrait les polygones (points) de tous les <polygon> et <path> (si possible) d'un SVG.
    Retourne une liste de np.array de points (N,2).
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    polygons = []
    for elem in root.iter():
        tag = elem.tag.split('}')[-1]
        if tag == 'polygon' and 'points' in elem.attrib:
            pts = [tuple(map(float, p.split(','))) for p in elem.attrib['points'].strip().split()]
            polygons.append(np.array(pts))
        elif tag == 'path' and 'd' in elem.attrib:
            try:
                from svgpathtools import parse_path
                path = parse_path(elem.attrib['d'])
                pts = [path.point(t) for t in np.linspace(0, 1, 100)]
                pts = np.array([[pt.real, pt.imag] for pt in pts])
                polygons.append(pts)
            except ImportError:
                pass
    return polygons

def procrustes_registration(A, B):
    """
    Aligne B sur A (translation, rotation, scale) via Procrustes (minimise la distance quadratique).
    A et B : np.array (N,2) (même nombre de points, ou B sera rééchantillonné)
    Retourne B_aligned, la disparité.
    """
    if len(A) != len(B):
        idxs = np.linspace(0, len(B)-1, len(A)).astype(int)
        B = B[idxs]
    mtx1, mtx2, disparity = procrustes(A, B)
    return mtx1, mtx2, disparity

def compare_svg_shapes_registration(svg1, svg2):
    """
    Compare les polygones extraits de deux SVG, tente une registration (Procrustes) pour chaque couple,
    et affiche le décalage, l'échelle et la rotation nécessaires pour les superposer.
    """
    polys1 = extract_svg_polygons(svg1)
    polys2 = extract_svg_polygons(svg2)
    print(f"[compare_svg_shapes_registration] {len(polys1)} polygones dans {svg1}, {len(polys2)} dans {svg2}")
    for i, poly1 in enumerate(polys1):
        if i >= len(polys2):
            break
        poly2 = polys2[i]
        mtx1, mtx2, disparity = procrustes_registration(poly1, poly2)
        print(f"Polygone {i}: Disparity (erreur quadratique) = {disparity:.6f}")
        c1 = poly1.mean(axis=0)
        c2 = poly2.mean(axis=0)
        # Calculer la forme alignée à partir de la matrice de transformation Procrustes
        aligned = mtx2 * poly2.std(axis=0)
        try:
            plt.figure()
            plt.plot(poly1[:,0], poly1[:,1], 'o-', label='SVG1')
            plt.plot(poly2[:,0], poly2[:,1], 'x--', label='SVG2')
            plt.plot(aligned[:,0], aligned[:,1], 's-', label='SVG2 aligné')
            plt.legend()
            plt.title(f'Polygone {i} registration')
            plt.show()
        except ImportError:
            pass
