import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
import cairosvg
from PIL import Image
from io import BytesIO

# Rasterise les deux SVGs sur une grille de pixels
def svg_to_array_and_save(svg_path, out_path, size=256):
    """
    Convertit un SVG en image numpy binaire (noir/blanc) et sauvegarde le PNG avec fond blanc.
    - svg_path : chemin du SVG à rasteriser
    - out_path : chemin du PNG à sauvegarder
    - size : taille de la grille (pixels)
    """
    # Rasterise le SVG en PNG (cairosvg)
    png_bytes = cairosvg.svg2png(url=svg_path, output_width=size, output_height=size, background_color='white')
    # Ouvre l'image PNG en mode RGB
    img = Image.open(BytesIO(png_bytes)).convert('RGB')
    # Crée une image blanche de fond
    bg = Image.new('RGB', img.size, (255, 255, 255))
    # Superpose le rendu SVG sur le fond blanc
    img = Image.alpha_composite(bg.convert('RGBA'), img.convert('RGBA')).convert('L')
    img.save(out_path)
    arr = np.array(img)
    # Binarise (seuil à 200 pour éviter le bruit)
    arr_bin = (arr < 200).astype(np.uint8)
    return arr_bin

# --- Vérification des attributs viewBox, width, height des deux SVG ---
def print_svg_attrs(svg_path, label):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    viewBox = root.attrib.get('viewBox', None)
    width = root.attrib.get('width', None)
    height = root.attrib.get('height', None)
    print(f"[{label}] viewBox: {viewBox}, width: {width}, height: {height}")
    return viewBox, width, height

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
