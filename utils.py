import re
import os
import cairosvg
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from svgpathtools import parse_path, Path as Svg_Path, Line, CubicBezier, QuadraticBezier, Arc
from PIL import Image
from io import BytesIO

def parse_transform(transform_str):
    """
    Parse une chaîne de transformation SVG (matrix, translate, scale, etc.) et retourne la matrice 3x3.
    Seul 'matrix(a,b,c,d,e,f)' est géré pour l'instant.
    """
    if not transform_str:
        return np.eye(3)
    m = re.search(r"matrix\(([^)]+)\)", transform_str)
    if m:
        vals = [float(v) for v in m.group(1).split(",")]
        if len(vals) == 6:
            a, b, c, d, e, f = vals
            mat = np.array([
                [a, c, e],
                [b, d, f],
                [0, 0, 1]
            ])
            return mat
    # Si pas de matrix, retourne identité
    return np.eye(3)

def apply_matrix_to_point(x, y, mat):
    pt = np.array([x, y, 1])
    res = mat @ pt
    return res[0], res[1]

def transform_path_d(d, mat):
    path = parse_path(d)
    new_path = Svg_Path()
    for seg in path:
        if isinstance(seg, Line):
            start = apply_matrix_to_point(seg.start.real, seg.start.imag, mat)
            end = apply_matrix_to_point(seg.end.real, seg.end.imag, mat)
            new_path.append(Line(complex(*start), complex(*end)))
        elif isinstance(seg, CubicBezier):
            start = apply_matrix_to_point(seg.start.real, seg.start.imag, mat)
            control1 = apply_matrix_to_point(seg.control1.real, seg.control1.imag, mat)
            control2 = apply_matrix_to_point(seg.control2.real, seg.control2.imag, mat)
            end = apply_matrix_to_point(seg.end.real, seg.end.imag, mat)
            new_path.append(CubicBezier(complex(*start), complex(*control1), complex(*control2), complex(*end)))
        elif isinstance(seg, QuadraticBezier):
            start = apply_matrix_to_point(seg.start.real, seg.start.imag, mat)
            control = apply_matrix_to_point(seg.control.real, seg.control.imag, mat)
            end = apply_matrix_to_point(seg.end.real, seg.end.imag, mat)
            new_path.append(QuadraticBezier(complex(*start), complex(*control), complex(*end)))
        elif isinstance(seg, Arc):
            # Pour les arcs, il faut aussi transformer les rayons et l'angle, ici on ne fait que translater les points
            start = apply_matrix_to_point(seg.start.real, seg.start.imag, mat)
            end = apply_matrix_to_point(seg.end.real, seg.end.imag, mat)
            new_path.append(Line(complex(*start), complex(*end)))  # approximation
        else:
            new_path.append(seg)
    return new_path.d()

def flatten_svg_transforms(svg_file, debug_dir=None):
    # Détermination du fichier SVG d'entrée à partir des règles données
    svg_file_in = None
    svg_file_base_name = None
    svg_file_path = str(svg_file)
    if svg_file_path.endswith("normalized.svg"):
        # On prend juste le nom de base sans le suffixe _normalized
        svg_file_base_name = os.path.splitext(os.path.basename(svg_file_path))[0].replace("_normalized", "")
        svg_file_in = svg_file_path
    elif debug_dir is not None and os.path.isdir(debug_dir):
        # Cherche un fichier contenant "normalized" dans debug_dir
        for fname in os.listdir(debug_dir):
            if "normalized" in fname and fname.endswith(".svg"):
                svg_file_in = os.path.join(debug_dir, fname)
                svg_file_base_name = os.path.splitext(fname)[0].replace("_normalized", "")
                break
        if svg_file_in is None:
            raise FileNotFoundError("Aucun fichier SVG 'normalized' trouvé dans le dossier debug_dir.")
    else:
        raise FileNotFoundError("Impossible de déterminer le fichier SVG d'entrée pour flatten_svg_transforms.")

    tree = ET.parse(svg_file_in)
    root = tree.getroot()

    def recursive_apply(elem, parent_mat=np.eye(3)):
        # Récupère la matrice de l'élément courant
        mat = parse_transform(elem.attrib.get('transform', None)) @ parent_mat
        # Applique la matrice aux enfants
        for child in elem:
            recursive_apply(child, mat)
        # Applique la matrice aux paths
        if strip_namespace(elem.tag) == 'path' and 'd' in elem.attrib:
            elem.attrib['d'] = transform_path_d(elem.attrib['d'], mat)
        # Supprime l'attribut transform
        if 'transform' in elem.attrib:
            del elem.attrib['transform']

    # Applique la transformation récursive à partir de la racine
    recursive_apply(root)


    # Force la balise racine à être <svg> sans namespace explicite
    root.tag = 'svg'
    # Réordonne les attributs pour que version et xmlns soient en premier
    ns = 'http://www.w3.org/2000/svg'
    root.attrib['version'] = '1.0'
    root.attrib['xmlns'] = ns
    def order_svg_attribs(attribs):
        ordered = [('version', attribs.get('version', '1.0')), ('xmlns', attribs.get('xmlns', ns))]
        for k, v in attribs.items():
            if k not in ('version', 'xmlns'):
                ordered.append((k, v))
        return ordered
    attribs_dict = dict(root.attrib)
    root.attrib.clear()
    for k, v in order_svg_attribs(attribs_dict):
        root.set(k, v)

    # Retire le namespace des balises enfants uniquement (pas la racine)
    for elem in root.iter():
        if elem is not root:
            elem.tag = strip_namespace(elem.tag)
            elem.attrib = {strip_namespace(k): v for k, v in elem.attrib.items()}

    # Détermine le nom du fichier de sortie
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        svg_file_out = os.path.join(debug_dir, f"{svg_file_base_name}_flattened.svg")
    else:
        svg_file_out = f"{svg_file_base_name}_flattened.svg"

    # Génère le SVG sans déclaration XML ni doctype
    normalized_svg = ET.tostring(root, encoding='unicode', method='xml')
    normalized_svg = re.sub(r'<\?xml[^>]*>\s*', '', normalized_svg)
    normalized_svg = re.sub(r'<!DOCTYPE[^>]*>\s*', '', normalized_svg)
    with open(svg_file_out, 'w', encoding='utf-8') as f:
        f.write(normalized_svg)

    return svg_file_out

def find_parent(root, child):
    for parent in root.iter():
        if child in list(parent):
            return parent
    return None

def write_svg_to_file(svg, filepath):
    tmp_file_path = os.path.splitext(filepath)[0] + "_normalized.svg"
    with open(tmp_file_path, "w") as f:
        f.write(svg)
    return tmp_file_path

def strip_namespace(tag):
    """
    Supprime le namespace XML d'un tag (ex : '{http://www.w3.org/2000/svg}path' -> 'path').
    Args:
        tag (str): Tag XML potentiellement avec namespace.
    Returns:
        str: Tag sans namespace.
    """
    return re.sub(r'\{.*\}', '', tag)

def propagate_attributes(elem, inherited=None):
    """
    Propage les attributs SVG (fill, stroke) hérités dans l'arbre XML.
    Args:
        elem: Élément XML courant.
        inherited: Dictionnaire des attributs hérités.
    """
    if inherited is None:
        inherited = {}
    current = inherited.copy()
    for attr in ['fill', 'stroke']:
        if attr in elem.attrib:
            current[attr] = elem.attrib.pop(attr)
    for child in list(elem):
        propagate_attributes(child, current)
    if elem.tag == 'path':
        for attr, value in current.items():
            if attr not in elem.attrib:
                elem.attrib[attr] = value

def normalize_svg_fill(svg_file_path, debug_dir=None):
    tree = ET.parse(svg_file_path)
    root = tree.getroot()

    ns = 'http://www.w3.org/2000/svg'
    # Forcer la balise racine <svg> à avoir version="1.0" et xmlns en premier
    root.tag = 'svg'  # retire le namespace du tag racine
    root.attrib['version'] = '1.0'
    root.attrib['xmlns'] = ns

    # Réordonner les attributs pour que version et xmlns soient en premier
    def order_svg_attribs(attribs):
        ordered = [('version', attribs.get('version', '1.0')), ('xmlns', attribs.get('xmlns', ns))]
        for k, v in attribs.items():
            if k not in ('version', 'xmlns'):
                ordered.append((k, v))
        return ordered
    # On efface et on remet les attributs dans l'ordre voulu
    attribs_dict = dict(root.attrib)
    root.attrib.clear()
    for k, v in order_svg_attribs(attribs_dict):
        root.set(k, v)

    # On retire le namespace des balises enfants uniquement (pas la racine)
    for elem in root.iter():
        if elem is not root:
            elem.tag = strip_namespace(elem.tag)
            elem.attrib = {strip_namespace(k): v for k, v in elem.attrib.items()}

    propagate_attributes(root)

    for bad_tag in ['metadata', 'desc', 'title']:
        for elem in root.findall(f'.//{bad_tag}'):
            parent = find_parent(root, elem)
            if parent is not None:
                parent.remove(elem)

    # Générer le SVG sans déclaration XML ni doctype
    normalized_svg = ET.tostring(root, encoding='unicode', method='xml')
    # Supprimer toute déclaration XML ou doctype si présente
    normalized_svg = re.sub(r'<\?xml[^>]*>\s*', '', normalized_svg)
    normalized_svg = re.sub(r'<!DOCTYPE[^>]*>\s*', '', normalized_svg)

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(svg_file_path))[0]
        normd_file_path = os.path.join(debug_dir, f"{base_name}_normalized.svg")
        with open(normd_file_path, "w", encoding="utf-8") as f:
            f.write(normalized_svg)
    else:
        normd_file_path = write_svg_to_file(normalized_svg, svg_file_path)
    return normd_file_path

def extract_subpaths(path, sampling):
    """
    Découpe un objet svgpathtools.Path (importé as Svg_Path) en sous-chemins (subpaths) selon les discontinuités.
    Args:
        path: Objet svgpathtools.Path (importé as Svg_Path)
        sampling: Nombre de points d'échantillonnage par segment
    Returns:
        list: Liste de sous-listes de points [x, y]
    """
    subpaths = []
    current_subpath = []
    prev_end = None
    for segment in path:
        if prev_end is not None and segment.start != prev_end:
            if current_subpath:
                subpaths.append(current_subpath)
            current_subpath = []
        pts = [segment.point(t) for t in np.linspace(0, 1, sampling)]
        for pt in pts:
            current_subpath.append([pt.real, pt.imag])
        prev_end = segment.end
    if current_subpath:
        subpaths.append(current_subpath)
    return subpaths

def rdp(points, epsilon=0.2):
    """
    Simplifie une liste de points 2D par l'algorithme de Ramer-Douglas-Peucker.
    Args:
        points: Liste de points [x, y]
        epsilon: Tolérance de simplification
    Returns:
        list: Liste de points simplifiés
    """
    if len(points) < 3:
        return points
    from math import hypot
    def point_line_dist(pt, start, end):
        if start == end:
            return hypot(pt[0] - start[0], pt[1] - start[1])
        else:
            n = abs((end[0]-start[0])*(start[1]-pt[1]) - (start[0]-pt[0])*(end[1]-start[1]))
            d = hypot(end[0]-start[0], end[1]-start[1])
            return n/d
    dmax = 0.0
    index = 0
    for i in range(1, len(points)-1):
        d = point_line_dist(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax > epsilon:
        rec1 = rdp(points[:index+1], epsilon)
        rec2 = rdp(points[index:], epsilon)
        return rec1[:-1] + rec2
    else:
        return [points[0], points[-1]]

def filter_points(points):
    """
    Filtre les points trop proches pour éviter les doublons dans une polyline.
    Args:
        points: Liste de points [x, y]
    Returns:
        list: Liste filtrée de points [x, y]
    """
    filtered = []
    if points:
        filtered.append(points[0])
        for j in range(1, len(points)):
            p1 = np.array(filtered[-1])
            p2 = np.array(points[j])
            if np.linalg.norm(p2 - p1) > 1e-6:
                filtered.append(points[j])
    return filtered

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
