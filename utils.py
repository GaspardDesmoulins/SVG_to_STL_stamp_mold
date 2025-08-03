import re
import os
import logging
import cairosvg
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from shapely.geometry import Polygon as ShapelyPolygon
from svgpathtools import parse_path, Path as Svg_Path, Line, CubicBezier, QuadraticBezier, Arc
from PIL import Image
from io import BytesIO
from shapely.geometry import Polygon

def strip_namespace(tag):
    """
    Supprime le namespace XML d'un tag (ex : '{http://www.w3.org/2000/svg}path' -> 'path').
    Args:
        tag (str): Tag XML potentiellement avec namespace.
    Returns:
        str: Tag sans namespace.
    """
    return re.sub(r'\{.*\}', '', tag)

def flatten_tree(elem, parent_mat=np.eye(3), root=None):
    """
    Parcours récursif de l'arbre XML SVG.
    Pour chaque élément :
    - Calcule la matrice de transformation cumulée (parent_mat * transform local)
    - Applique la transformation aux enfants en premier (feuilles -> racine)
    - Si l'élément est un <path>, applique la matrice cumulée à ses coordonnées
    - Supprime l'attribut 'transform' après application
    - Si l'élément est un groupe <g>, remonte ses enfants dans le parent et supprime le groupe
    """
    if root is None:
        svgroot = elem
    else:
        svgroot = root
    # Calcule la matrice cumulée pour cet élément
    mat = parse_transform(elem.attrib.get('transform', None)) @ parent_mat
    # On traite d'abord les enfants (feuilles les plus éloignées)
    children = list(elem)
    for child in children:
        flatten_tree(child, mat, root=svgroot)
    # Si c'est un path, on applique la matrice cumulée
    if strip_namespace(elem.tag) == 'path' and 'd' in elem.attrib:
        elem.attrib['d'] = transform_path_d(elem.attrib['d'], mat)
    # On retire l'attribut transform
    if 'transform' in elem.attrib:
        del elem.attrib['transform']
    # Si c'est un groupe <g>, on "aplatit" en remontant ses enfants dans le parent
    if strip_namespace(elem.tag) == 'g':
        parent = find_parent(root, elem)
        if parent is not None:
            idx = list(parent).index(elem)
            for child in list(elem):
                parent.insert(idx, child)
                idx += 1
            parent.remove(elem)

def convert_rect_ellipse_to_path(root):
    """
    Convertit tous les <rect> et <ellipse> du SVG en <path> équivalents (en place dans l'arbre XML).
    """
    import math
    # --- Conversion des rectangles ---
    # On collecte tous les <rect> à convertir, puis on prépare les remplacements par parent
    rects_to_convert = [rect for rect in root.iter() if strip_namespace(rect.tag) == 'rect']
    parent_to_rects = {}
    for rect in rects_to_convert:
        parent = find_parent(root, rect)
        if parent is not None:
            parent_to_rects.setdefault(parent, []).append(rect)
    # Pour chaque parent, on ajoute les nouveaux <path> à la fin, puis on supprime tous les <rect>
    for parent, rects in parent_to_rects.items():
        new_paths = []
        for rect in rects:
            # Extraction des attributs principaux
            x = float(rect.attrib.get('x', 0))
            y = float(rect.attrib.get('y', 0))
            w = float(rect.attrib.get('width', 0))
            h = float(rect.attrib.get('height', 0))
            rx = float(rect.attrib.get('rx', 0)) if 'rx' in rect.attrib else 0
            ry = float(rect.attrib.get('ry', 0)) if 'ry' in rect.attrib else 0
            stroke = rect.attrib.get('stroke', None)
            # stroke-width : si défini, on prend la valeur, sinon 1 si stroke présent, sinon 0
            if stroke is not None:
                if 'stroke-width' in rect.attrib:
                    stroke_width = float(rect.attrib['stroke-width'])
                else:
                    stroke_width = 1.0
            else:
                stroke_width = 0
            # Si rx ou ry > 0, on ignore pour l'instant (arrondis non gérés)
            if rx > 0 or ry > 0:
                continue
            # Si stroke et stroke-width > 0, on élargit simplement le rectangle vers l'extérieur de sw/2
            if stroke and stroke_width > 0:
                sw = stroke_width
                x0, y0 = x - sw/2, y - sw/2
                x1, y1 = x + w + sw/2, y - sw/2
                x2, y2 = x + w + sw/2, y + h + sw/2
                x3, y3 = x - sw/2, y + h + sw/2
                d = f"M {x0},{y0} L {x1},{y1} L {x2},{y2} L {x3},{y3} Z"
            else:
                # Génère explicitement les 4 coins pour éviter les imprécisions, sans répéter le point de départ
                x0, y0 = x, y
                x1, y1 = x + w, y
                x2, y2 = x + w, y + h
                x3, y3 = x, y + h
                d = f"M {x0},{y0} L {x1},{y1} L {x2},{y2} L {x3},{y3} Z"
            # Création du nouvel élément <path> avec les mêmes attributs (copie)
            path_elem = ET.Element('path', dict(rect.attrib))
            path_elem.attrib['d'] = d
            # Supprime les attributs spécifiques au rect
            for k in ['x', 'y', 'width', 'height', 'rx', 'ry', 'stroke', 'stroke-width']:
                if k in path_elem.attrib:
                    del path_elem.attrib[k]
            # Ajoute un attribut fill explicite (par défaut noir)
            path_elem.attrib['fill'] = path_elem.attrib.get('fill', '#000')
            new_paths.append(path_elem)
        # Ajoute tous les nouveaux paths à la fin du parent
        for path_elem in new_paths:
            parent.append(path_elem)
        # Supprime tous les <rect> du parent
        for rect in rects:
            parent.remove(rect)

    # --- Conversion des ellipses ---
    # On collecte tous les <ellipse> à convertir, puis on prépare les remplacements par parent
    ellipses_to_convert = [ellipse for ellipse in root.iter() if strip_namespace(ellipse.tag) == 'ellipse']
    parent_to_ellipses = {}
    for ellipse in ellipses_to_convert:
        parent = find_parent(root, ellipse)
        if parent is not None:
            parent_to_ellipses.setdefault(parent, []).append(ellipse)
    # Pour chaque parent, on ajoute les nouveaux <path> à la fin, puis on supprime tous les <ellipse>
    for parent, ellipses in parent_to_ellipses.items():
        new_paths = []
        for ellipse in ellipses:
            # Extraction des attributs principaux
            cx = float(ellipse.attrib.get('cx', 0))
            cy = float(ellipse.attrib.get('cy', 0))
            rx = float(ellipse.attrib.get('rx', 0))
            ry = float(ellipse.attrib.get('ry', 0))
            stroke = ellipse.attrib.get('stroke', None)
            # stroke-width : si défini, on prend la valeur, sinon 1 si stroke présent, sinon 0
            if stroke is not None:
                if 'stroke-width' in ellipse.attrib:
                    stroke_width = float(ellipse.attrib['stroke-width'])
                else:
                    stroke_width = 1.0
            else:
                stroke_width = 0
            # Si stroke, on élargit l'ellipse
            if stroke and stroke_width > 0:
                rx2 = rx + stroke_width/2
                ry2 = ry + stroke_width/2
            else:
                rx2 = rx
                ry2 = ry
            # Approximation de l'ellipse par 4 courbes de Bézier cubiques (méthode de Spiro/Adobe)
            # Constante magique pour l'approximation (voir https://spencermortensen.com/articles/bezier-circle/)
            kappa = 0.5522847498307936
            dx = rx2 * kappa
            dy = ry2 * kappa
            # Points de contrôle
            p0 = (cx + rx2, cy)
            c1 = (cx + rx2, cy + dy)
            c2 = (cx + dx, cy + ry2)
            p1 = (cx, cy + ry2)
            c3 = (cx - dx, cy + ry2)
            c4 = (cx - rx2, cy + dy)
            p2 = (cx - rx2, cy)
            c5 = (cx - rx2, cy - dy)
            c6 = (cx - dx, cy - ry2)
            p3 = (cx, cy - ry2)
            c7 = (cx + dx, cy - ry2)
            c8 = (cx + rx2, cy - dy)
            # Construction du path d
            d = (
                f"M {p0[0]},{p0[1]} "
                f"C {c1[0]},{c1[1]} {c2[0]},{c2[1]} {p1[0]},{p1[1]} "
                f"C {c3[0]},{c3[1]} {c4[0]},{c4[1]} {p2[0]},{p2[1]} "
                f"C {c5[0]},{c5[1]} {c6[0]},{c6[1]} {p3[0]},{p3[1]} "
                f"C {c7[0]},{c7[1]} {c8[0]},{c8[1]} {p0[0]},{p0[1]} Z"
            )
            # Création du nouvel élément <path> avec les mêmes attributs (copie)
            path_elem = ET.Element('path', dict(ellipse.attrib))
            path_elem.attrib['d'] = d
            # Supprime les attributs spécifiques à ellipse
            for k in ['cx', 'cy', 'rx', 'ry', 'stroke', 'stroke-width']:
                if k in path_elem.attrib:
                    del path_elem.attrib[k]
            # Ajoute un attribut fill explicite (par défaut noir)
            path_elem.attrib['fill'] = path_elem.attrib.get('fill', '#000')
            new_paths.append(path_elem)
        # Ajoute tous les nouveaux paths à la fin du parent
        for path_elem in new_paths:
            parent.append(path_elem)
        # Supprime tous les <ellipse> du parent
        for ellipse in ellipses:
            parent.remove(ellipse)

# --- Calcul de l'aire signée pour orientation du polygone ---
def polygon_signed_area(points):
    """
    Calcule l'aire signée d'un polygone (x, y).
    Aire > 0 : polygone orienté anti-horaire (extérieur à gauche)
    Aire < 0 : polygone horaire (extérieur à droite)
    """
    area = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        area += (x0 * y1) - (x1 * y0)
    return area / 2.0

def offset_polygon_along_normals(points, offset_distance, centroid_direction, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    # Filtre les points dupliqués consécutifs pour éviter les artefacts
    filtered_points = [points[0]]
    for pt in points[1:]:
        if np.linalg.norm(np.array(pt) - np.array(filtered_points[-1])) > 1e-10:
            filtered_points.append(pt)
    points = filtered_points
    pts = np.array(points)
    num_points = len(pts)
    # S'assurer que le polygone est fermé (premier et dernier point identiques)
    if not np.allclose(pts[0], pts[-1]):
        closed_pts = np.vstack([pts, pts[0]])
    else:
        closed_pts = pts
    # Détermination de l'orientation du polygone (aire signée)
    area = polygon_signed_area(points)
    is_ccw = area > 0  # True si anti-horaire

    # Correction d'orientation : outer doit être CCW, inner doit être CW
    # On inverse les points si besoin pour garantir l'orientation attendue
    if is_ccw:
        closed_pts = np.flipud(closed_pts)

    # Vérification initiale : si le polygone d'origine n'est pas simple, on ne tente pas d'offset
    if not ShapelyPolygon(points).is_simple:
        logger.warning("Polygone d'origine non simple, offset ignoré.")
        return list(points) + [points[0]]

    min_offset = 1e-6
    max_iter = 16
    offset_points = [None] * num_points
    for i in range(num_points):
        prev_point = closed_pts[(i - 1) % num_points]
        current_point = closed_pts[i]
        next_point = closed_pts[(i + 1) % num_points]
        # Calcul des vecteurs des segments adjacents
        vec_prev = current_point - prev_point
        vec_next = next_point - current_point
        # Calcul des normales locales aux segments
        normal_prev = np.array([-vec_prev[1], vec_prev[0]])
        normal_next = np.array([-vec_next[1], vec_next[0]])
        norm_prev = np.linalg.norm(normal_prev)
        norm_next = np.linalg.norm(normal_next)
        if norm_prev < 1e-10 or norm_next < 1e-10:
            # Cas dégénéré : un des segments est trop court
            normal = normal_prev if norm_next < 1e-10 else normal_next
            if np.linalg.norm(normal) < 1e-10:
                normal = np.array([1.0, 0.0])
        else:
            # Moyenne des normales pour lisser la direction locale
            normal_prev /= norm_prev
            normal_next /= norm_next
            normal = (normal_prev + normal_next)
            if np.linalg.norm(normal) < 1e-8:
                normal = normal_prev
            else:
                normal /= np.linalg.norm(normal)
        offset_local = float(offset_distance)
        for attempt in range(max_iter):
            # Offset toujours dans le sens centroid_direction
            # Cela garantit que l'outer est toujours vers l'extérieur et l'inner vers l'intérieur,
            # indépendamment de l'orientation initiale du polygone.
            final_offset = offset_local * centroid_direction
            candidate_point = tuple(current_point + final_offset * normal)
            # On teste le polygone avec ce point déplacé
            test_points = [offset_points[j] if (offset_points[j] is not None) else tuple(closed_pts[j]) for j in range(num_points)]
            test_points[i] = candidate_point
            test_points_closed = test_points + [test_points[0]]
            poly = ShapelyPolygon(test_points_closed)
            if poly.is_simple:
                offset_points[i] = candidate_point
                break
            else:
                # Si auto-intersection, on réduit l'offset localement
                offset_local *= 0.5
                if abs(offset_local) < min_offset:
                    logger.warning(f"Offset trop faible pour le point {i}, auto-intersection persistante.")
                    offset_points[i] = tuple(current_point)
                    break
    # On ferme le polygone
    offset_points.append(offset_points[0])
    return offset_points

def align_resampled_to_reference(resampled, reference):
    """
    Décale circulairement et inverse éventuellement le sens de resampled pour minimiser la distance à reference.
    """
    ref = np.array(reference)
    res = np.array(resampled)
    # Teste les deux sens
    best = None
    best_dist = None
    for direction in [1, -1]:
        if direction == -1:
            res_dir = res[::-1]
        else:
            res_dir = res
        # Cherche le meilleur décalage circulaire
        for shift in range(len(res_dir)):
            res_shift = np.roll(res_dir, -shift, axis=0)
            dist = np.sum(np.linalg.norm(res_shift - ref, axis=1))
            if best is None or dist < best_dist:
                best = res_shift
                best_dist = dist
    return [tuple(pt) for pt in best]

def is_visible(attr, force_all_contours=False):
    if force_all_contours:
        return True
    fill = attr.get('fill', None)
    stroke = attr.get('stroke', None)
    def is_none_or_transparent(val):
        if val is None:
            return False
        val = val.strip().lower()
        return val in ['none', 'transparent']
    return not (is_none_or_transparent(fill) and is_none_or_transparent(stroke))


def resample_polygon(points, n):
    """
    Rééchantillonne une liste de points (fermée) pour obtenir n points régulièrement espacés.
    """
    import numpy as np
    pts = np.array(points)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumlen = np.concatenate([[0], np.cumsum(dists)])
    total_length = cumlen[-1]
    if total_length == 0 or n < 2:
        return [tuple(pt) for pt in pts[:-1]]
    new_locs = np.linspace(0, total_length, n+1)[:-1]
    new_pts = []
    for loc in new_locs:
        idx = np.searchsorted(cumlen, loc, side='right') - 1
        if idx >= len(pts)-1:
            idx = len(pts)-2
        seg_start = pts[idx]
        seg_end = pts[idx+1]
        seg_len = cumlen[idx+1] - cumlen[idx]
        if seg_len == 0:
            new_pt = seg_start
        else:
            t = (loc - cumlen[idx]) / seg_len
            new_pt = seg_start + t * (seg_end - seg_start)
        new_pts.append(tuple(new_pt))
    return new_pts

def extract_outer_inners_groups_from_svg_paths(root):
    """
    Pour chaque path du SVG, découpe en subpaths, identifie l'outer principal et les inners associés selon la règle even-odd.
    Retourne une liste de groupes {'outer': {...}, 'inners': [...]}, chaque groupe correspondant à un ensemble outer/inners.
    """
    # --- Nouvelle logique : arbre d'inclusion et regroupement par outers croissants ---
    groups = []
    path_elems = [elem for elem in root.iter() if elem.tag == 'path' and 'd' in elem.attrib]
    for elem in path_elems:
        try:
            path = parse_path(elem.attrib['d'])
            subpaths = extract_subpaths(path, nb_pts_smpl=3)
            # pas de simplification RDP pour préserver la douceur des courbes (notamment les ellipses)
            subpaths = [sp for sp in subpaths if len(sp) > 2]
            
            if not subpaths:
                continue
            # 1. Construction de l'arbre d'inclusion
            n = len(subpaths)
            parents = [None] * n
            areas = [Polygon(sp).area for sp in subpaths]
            for i, pts_i in enumerate(subpaths):
                poly_i = Polygon(pts_i)
                min_area = None
                parent_idx = None
                for j, pts_j in enumerate(subpaths):
                    if i == j:
                        continue
                    poly_j = Polygon(pts_j)
                    # On vérifie que poly_j contient tous les points de poly_i (et pas l'inverse)
                    # On utilise contains(poly_i) pour éviter les cycles
                    if poly_j.contains(poly_i):
                        area_j = poly_j.area
                        if min_area is None or area_j < min_area:
                            min_area = area_j
                            parent_idx = j
                parents[i] = parent_idx
            # 2. Calcul du niveau d'imbrication (profondeur)
            levels = [0] * n
            for i in range(n):
                idx = i
                while parents[idx] is not None:
                    levels[i] += 1
                    idx = parents[idx]
            # 3. Regroupement des outers (niveau pair) en commençant par les plus petits
            outers = [i for i in range(n) if levels[i] % 2 == 0]
            # Trie les outers par aire croissante
            outers_sorted = sorted(outers, key=lambda idx: areas[idx])
            # 4. Pour chaque outer, regroupe ses inners directs (niveau impair, parent = outer)
            for outer_idx in outers_sorted:
                inners_idx = [i for i in range(n) if parents[i] == outer_idx and levels[i] == levels[outer_idx] + 1]
                group = {
                    'outer': {'elem': elem, 'pts': subpaths[outer_idx], 'd': None},
                    'inners': [{'pts': subpaths[i], 'd': None} for i in inners_idx]
                }
                # Génère le d pour chaque subpath (outer et inners)
                def points_to_d(pts):
                    # Ne pas répéter le point de départ à la fin
                    n = len(pts)
                    if n < 2:
                        return ''
                    path_obj = Svg_Path()
                    for k in range(n-1):
                        start = complex(*pts[k])
                        end = complex(*pts[k+1])
                        path_obj.append(Line(start, end))
                    # Ferme le chemin si ce n'est pas déjà le cas
                    if not (np.allclose(pts[0], pts[-1])):
                        path_obj.append(Line(complex(*pts[-1]), complex(*pts[0])))
                    return path_obj.d()
                group['outer']['d'] = points_to_d(group['outer']['pts'])
                for idx, inner in enumerate(group['inners']):
                    inner['d'] = points_to_d(inner['pts'])
                groups.append(group)
        except Exception:
            continue
    groups.reverse()
    return groups

def icp_affine(src_points, tgt_points, max_iter=2000, tol=1e-6):
    """
    Calcule la matrice affine optimale (3x3) qui transforme src_points en tgt_points (points appariés).
    Retourne les points alignés et la matrice de transformation.
    """
    src = src_points.copy()
    tgt = tgt_points.copy()
    # Ajoute la colonne homogène
    src_h = np.hstack([src, np.ones((src.shape[0], 1))])
    tgt_h = np.hstack([tgt, np.ones((tgt.shape[0], 1))])
    # Résolution par moindres carrés
    H, _, _, _ = np.linalg.lstsq(src_h, tgt_h, rcond=None)
    H = H.T
    # Applique la transformation
    src_aligned = (H @ src_h.T).T[:, :2]
    return src_aligned, H

# Applique la matrice ICP à chaque polygone du SVG normalisé et génère un nouveau SVG
def apply_homography_to_svg_points_per_polygon(svg_in, svg_out, H_list):
    """
    Applique à chaque polygone du SVG la matrice affine ICP correspondante (H_list).
    - svg_in : chemin du SVG normalisé
    - svg_out : chemin du SVG de sortie
    - H_list : liste de matrices (une par polygone)
    """

    def apply_matrix_to_point(x, y, mat):
        pt = np.array([x, y, 1])
        res = mat @ pt
        return res[0], res[1]

    def transform_path_d_per_poly(d, mat):
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
                start = apply_matrix_to_point(seg.start.real, seg.start.imag, mat)
                end = apply_matrix_to_point(seg.end.real, seg.end.imag, mat)
                new_path.append(Line(complex(*start), complex(*end)))  # approximation
            else:
                new_path.append(seg)
        return new_path.d()

    tree = ET.parse(svg_in)
    root = tree.getroot()

    flatten_tree(root)

    # Applique la matrice d'homographie sur chaque path (polygone)
    path_elems = [elem for elem in root.iter() if strip_namespace(elem.tag) == 'path' and 'd' in elem.attrib]
    if len(path_elems) != len(H_list):
        print(f"Attention : {len(path_elems)} paths trouvés, {len(H_list)} matrices fournies.")
    n = min(len(path_elems), len(H_list))
    for i in range(n):
        elem = path_elems[i]
        mat = H_list[i]
        elem.attrib['d'] = transform_path_d_per_poly(elem.attrib['d'], mat)

    # Force la balise racine à être <svg> sans namespace explicite
    root.tag = 'svg'
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

    # Sauvegarde le SVG modifié sans déclaration XML ni doctype
    normalized_svg = ET.tostring(root, encoding='unicode', method='xml')
    normalized_svg = re.sub(r'<\?xml[^>]*>\s*', '', normalized_svg)
    normalized_svg = re.sub(r'<!DOCTYPE[^>]*>\s*', '', normalized_svg)
    with open(svg_out, 'w', encoding='utf-8') as f:
        f.write(normalized_svg)

# Extraction et composition des matrices de transformation du SVG
def extract_svg_transform_matrices(svg_path):
    """
    Extrait toutes les matrices de transformation (transform="matrix(...)") du SVG et les compose.
    Retourne la matrice globale 3x3.
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(svg_path)
    root = tree.getroot()
    matrices = []
    for elem in root.iter():
        t = elem.attrib.get('transform', None)
        if t:
            mat = parse_transform(t)
            matrices.append(mat)
    # Compose les matrices dans l'ordre d'apparition (première appliquée en premier)
    if matrices:
        mat_global = np.eye(3)
        for m in matrices:
            mat_global = m @ mat_global
        return mat_global
    else:
        return np.eye(3)

def verify_affine_vs_svg_transform_per_polygon(svg_normalized, svg_flattened):
    """
    Pour chaque paire de polygones (svg_normalized, svg_flattened),
    calcule la matrice affine ICP et la compare à la matrice composée des balises transform du SVG normalisé.
    Affiche les deux matrices.
    """
    polys_norm = extract_svg_polygons(svg_normalized)
    polys_flat = extract_svg_polygons(svg_flattened)
    if not polys_norm or not polys_flat:
        print("Aucun polygone trouvé dans l'un des SVG.")
        return None
    H_svg = extract_svg_transform_matrices(svg_normalized)
    n = min(len(polys_norm), len(polys_flat))
    results = []
    for i in range(n):
        n_points = min(len(polys_norm[i]), len(polys_flat[i]))
        def resample_poly(poly, n):
            idxs = np.linspace(0, len(poly)-1, n).astype(int)
            return poly[idxs]
        poly_norm_rs = resample_poly(polys_norm[i], n_points)
        poly_flat_rs = resample_poly(polys_flat[i], n_points)
        _, H_icp = icp_affine(poly_norm_rs, poly_flat_rs)
        print(f"Polygone {i} :")
        print("  Matrice affine ICP (svg_normalized -> svg_flattened):\n", H_icp)
        print("  Matrice composée des balises transform du SVG normalisé:\n", H_svg)
        results.append((H_icp, H_svg))
    return results

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
    # On cherche le fichier normalisé dans le dossier de debug, ou on prend le fichier passé en argument
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
    # Sauvegarde des attributs globaux du SVG source (viewBox, width, height, etc.)
    original_attribs = dict(root.attrib)
    original_tag = root.tag

    flatten_tree(root)

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

def to_original_coords(pt, svg_scale, svg_min_x, svg_min_y):
            if svg_scale and svg_scale != 0:
                x = pt[0] / svg_scale + svg_min_x
                y = pt[1] / svg_scale + svg_min_y
                return (x, y)
            else:
                return pt

def normalize_svg_fill(svg_file_path, debug_dir=None):

    tree = ET.parse(svg_file_path)
    root = tree.getroot()

    # Conversion des rectangles et ellipses en paths avant toute opération
    convert_rect_ellipse_to_path(root)

    # --- Aplatissement des transformations SVG (logique de flatten_svg_transforms) ---
    flatten_tree(root)

    ns = 'http://www.w3.org/2000/svg'
    root.tag = 'svg'
    root.attrib['version'] = '1.0'
    root.attrib['xmlns'] = ns

    # Réordonner les attributs pour que version et xmlns soient en premier
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

    # --- Regroupement outer/inners via fonction dédiée ---
    all_groups = extract_outer_inners_groups_from_svg_paths(root)

    # On crée un nouveau SVG avec un path par groupe (outer + inners)
    new_root = ET.Element('svg', dict(root.attrib))
    # Copie viewBox, width, height si présents
    for attr in ['viewBox', 'width', 'height']:
        if attr in root.attrib:
            new_root.attrib[attr] = root.attrib[attr]

    for group in all_groups:
        # Crée un nouveau path pour l'outer
        outer_elem = group['outer']['elem']
        new_path = ET.Element('path', dict(outer_elem.attrib))
        # Concatène le d de l'outer et des inners
        d_concat = group['outer']['d']
        for inner in group['inners']:
            d_concat += ' ' + inner['d']
        new_path.attrib['d'] = d_concat
        new_root.append(new_path)

    # Générer le SVG sans déclaration XML ni doctype
    normalized_svg = ET.tostring(new_root, encoding='unicode', method='xml')
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

# Sur-échantillonnage adaptatif pour préserver la douceur des courbes de Bézier
def adaptive_bezier_sampling(segment, nb_pts_smpl, min_pts=10, max_pts=200, tol=0.1):
    # Pour les segments de Bézier, adapte le nombre de points à la courbure
    if isinstance(segment, (CubicBezier, QuadraticBezier)):
        # On estime la longueur de la courbe et la courbure max
        length = segment.length(error=1e-4)
        # On approxime la courbure max par la distance max entre la courbe et la corde
        ts = np.linspace(0, 1, 10)
        pts = np.array([[pt.real, pt.imag] for pt in [segment.point(t) for t in ts]])
        A = np.array([segment.point(0).real, segment.point(0).imag])
        B = np.array([segment.point(1).real, segment.point(1).imag])
        AB = B - A
        AP = pts - A
        norm_AB = np.linalg.norm(AB)
        if norm_AB < 1e-12:
            dists = np.linalg.norm(AP, axis=1)
        else:
            # Produit vectoriel 2D (résultat scalaire)
            cross = AB[0]*AP[:,1] - AB[1]*AP[:,0]
            dists = np.abs(cross) / norm_AB
        max_curv = np.max(dists)
        # Plus la courbure est forte, plus on met de points
        n = int(min(max_pts, max(min_pts, length * (1 + 10*max_curv) / tol)))
        ts = np.linspace(0, 1, n)
        return [segment.point(t) for t in ts]
    else:
        # Pour les segments linéaires ou arcs, sampling fixe
        return [segment.point(t) for t in np.linspace(0, 1, nb_pts_smpl)]


def extract_subpaths(path, nb_pts_smpl, bezier_sampling=False):
    """
    Découpe un objet svgpathtools.Path (importé as Svg_Path) en sous-chemins (subpaths) selon les discontinuités.
    Args:
        path: Objet svgpathtools.Path (importé as Svg_Path)
        nb_pts_smpl: Nombre de points d'échantillonnage par segment
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
        if bezier_sampling:
            pts = adaptive_bezier_sampling(segment, nb_pts_smpl, min_pts=10, max_pts=2000, tol=0.1)
        else:
            pts = [segment.point(t) for t in np.linspace(0, 1, nb_pts_smpl)]
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
        # Rééchantillonnage pour avoir le même nombre de points
        n_points = min(len(poly1), len(poly2))
        def resample_poly(poly, n):
            idxs = np.linspace(0, len(poly)-1, n).astype(int)
            return poly[idxs]
        poly1_rs = resample_poly(poly1, n_points)
        poly2_rs = resample_poly(poly2, n_points)
        # Recalage par ICP affine
        poly1_aligned, H = icp_affine(poly1_rs, poly2_rs)
        # Calcul de l'erreur moyenne après recalage
        error = np.mean(np.linalg.norm(poly1_aligned - poly2_rs, axis=1))
        print(f"Polygone {i}: Erreur moyenne après recalage ICP = {error:.6f}")
        print(f"Matrice de transformation affine (homographique) :\n{H}")
        try:
            plt.figure()
            plt.plot(poly1_rs[:,0], poly1_rs[:,1], 'o-', label='SVG1')
            plt.plot(poly2_rs[:,0], poly2_rs[:,1], 'x--', label='SVG2')
            plt.plot(poly1_aligned[:,0], poly1_aligned[:,1], 's-', label='SVG1 aligné')
            plt.legend()
            plt.title(f'Polygone {i} registration ICP')
            plt.show()
        except ImportError:
            pass
