import cadquery as cq
from svgpathtools import svg2paths2
import numpy as np
from pathlib import Path
import os
from xml.etree import ElementTree as ET
import re
import argparse
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


# === Paramètres du moule ===
MAX_DIMENSION = 100             # Dimension maximale (mm) du moule
MARGE = 2                       # Distance minimale entre motif et bordure
BASE_THICKNESS = 5             # Épaisseur de la base du moule
BORDER_HEIGHT = 4              # Hauteur des bordures du moule
BORDER_THICKNESS = 2           # Épaisseur des bordures
ENGRAVE_DEPTH = 2              # Profondeur d'extrusion du motif

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

def normalize_svg_fill(svg_file_path):
    tree = ET.parse(svg_file_path)
    root = tree.getroot()

    
    for elem in root.iter():
        elem.tag = strip_namespace(elem.tag)
        elem.attrib = {strip_namespace(k): v for k, v in elem.attrib.items()}

    propagate_attributes(root)

    for bad_tag in ['metadata', 'desc', 'title']:
        for elem in root.findall(f'.//{bad_tag}'):
            parent = find_parent(root, elem)
            if parent is not None:
                parent.remove(elem)

    normalized_svg = ET.tostring(root, encoding='unicode')
    normd_file_path = write_svg_to_file(normalized_svg, svg_file_path)
    return normd_file_path

def extract_subpaths(path, sampling):
    """
    Découpe un objet svgpathtools.Path en sous-chemins (subpaths) selon les discontinuités.
    Args:
        path: Objet svgpathtools.Path
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


def svg_to_cadquery_wires(svg_file, max_dimension, interactive=True):
    """
    Convertit un SVG en wires CadQuery, avec simplification et gestion de l'échelle.
    Retourne la liste des wires et l'historique des shapes.
    """
    svgfile = Path(svg_file)
    paths, attributes, svg_attr = svg2paths2(str(svgfile.absolute()))
    all_points = []
    shapes_data = []
    ignored_indices = []
    sampling_default = 30
    shape_history = {}
    shape_keys = []
    wire_to_shape = {}

    for path_idx, (path, attr) in enumerate(zip(paths, attributes)):
        fill = attr.get('fill', 'none')
        if fill in ['none', 'transparent']:
            continue
        print(f"\n--- Path {path_idx} ---")
        subpaths = extract_subpaths(path, sampling_default)
        print(f"Subpaths {len(subpaths)}")
        for sub_idx, sampled_points in enumerate(subpaths):
            if sampled_points:
                simple_points = rdp(sampled_points, epsilon=0.2)
                all_points.extend(simple_points)
                shapes_data.append(simple_points)
                shape_keys.append((path_idx, sub_idx))
                shape_history[(path_idx, sub_idx)] = {
                    'svg_path_idx': path_idx,
                    'svg_sub_idx': sub_idx,
                    'svg_attr': dict(attr),
                    'sampled_points': sampled_points,
                    'simplified_points': simple_points,
                    'cq_wire_index': None,
                }

    if not all_points:
        raise ValueError("No SVG paths found with fill to create mold.")

    xs = [pt[0] for pt in all_points]
    ys = [pt[1] for pt in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    svg_width = max_x - min_x
    svg_height = max_y - min_y
    if svg_width == 0 and svg_height == 0:
        scale = 1.0
    elif svg_width == 0:
        scale = (max_dimension - 2 * MARGE) / svg_height
    elif svg_height == 0:
        scale = (max_dimension - 2 * MARGE) / svg_width
    else:
        scale = (max_dimension - 2 * MARGE) / max(svg_width, svg_height)

    cq_wires = []
    for shape_idx, shape_points in enumerate(shapes_data):
        retry = True
        while retry:
            scaled_points = [((x - min_x) * scale, (y - min_y) * scale) for x, y in shape_points]
            filtered_points = filter_points(scaled_points)
            print(f"\nShape {shape_idx}: Nombre de points avant filtre: {len(shape_points)}, après filtre: {len(filtered_points)}")
            if len(filtered_points) > 2:
                try:
                    first_pt = np.array(filtered_points[0])
                    last_pt = np.array(filtered_points[-1])
                    dist = np.linalg.norm(first_pt - last_pt)
                    if dist > 1e-6:
                        filtered_points.append(tuple(filtered_points[0]))
                        print(f"Shape {shape_idx}: Fermeture forcée du polygone.")
                    # Tentative 1 : polyline().close()
                    temp_wp = cq.Workplane("XY").polyline(filtered_points).close()
                    if len(temp_wp.wires().objects) == 1:
                        single_wire = temp_wp.wires().objects[0]
                        verts = list(single_wire.Vertices())
                        if single_wire.isValid() and np.linalg.norm(np.array([verts[0].X, verts[0].Y]) - np.array([verts[-1].X, verts[-1].Y])) < 1e-6:
                            cq_wires.append(single_wire)
                            shape_history[shape_keys[shape_idx]]['cq_wire_index'] = len(cq_wires) - 1
                            wire_to_shape[len(cq_wires) - 1] = shape_keys[shape_idx]
                            print(f"Shape {shape_idx}: Wire valide créé (polyline).")
                            retry = False
                            continue
                    # Tentative 2 : makePolygon
                    try:
                        poly_wire = cq.Wire.makePolygon([cq.Vector(x, y) for x, y in filtered_points])
                        if poly_wire.isValid():
                            cq_wires.append(poly_wire)
                            shape_history[shape_keys[shape_idx]]['cq_wire_index'] = len(cq_wires) - 1
                            wire_to_shape[len(cq_wires) - 1] = shape_keys[shape_idx]
                            print(f"Shape {shape_idx}: Wire valide créé (makePolygon).")
                            retry = False
                            continue
                        else:
                            print(f"Shape {shape_idx}: makePolygon a produit un wire invalide.")
                    except Exception as e2:
                        print(f"Shape {shape_idx}: Erreur makePolygon: {e2}")
                except Exception as e:
                    print(f"Shape {shape_idx}: Erreur lors de la création du wire : {e}")
            else:
                print(f"Shape {shape_idx}: Pas assez de points pour créer un wire.")
            if retry and interactive:
                print(f"\n[INTERACTIF] Le polygone {shape_idx} n'a pas pu être converti en solide.")
                print("Options : [r] Relancer, [i] Ignorer, [q] Quitter.")
                user_choice = input(f"Que faire pour le polygone {shape_idx} ? (r/i/q) : ").strip().lower()
                if user_choice == 'r':
                    print("(Note : l'échantillonnage dynamique n'est pas encore implémenté pour ce polygone. Ignoré.)")
                    user_choice = 'i'
                if user_choice == 'i':
                    print(f"Polygone {shape_idx} ignoré.")
                    ignored_indices.append(shape_idx)
                    retry = False
                if user_choice == 'q':
                    print("Arrêt demandé par l'utilisateur.")
                    exit(0)
            else:
                if retry:
                    print(f"Polygone {shape_idx} ignoré automatiquement (mode non interactif).")
                    ignored_indices.append(shape_idx)
                retry = False
    shape_history['wire_to_shape'] = wire_to_shape
    return cq_wires, shape_history

def create_mold_base(svg_wires, margin, base_thickness, border_height, border_thickness, debug_dir, base_stl_name, export_base_stl=True):
    """
    Crée la base du moule (avec marge et bordure) à partir des wires SVG.
    Retourne le solide de base (mold) et le wire de contour utilisé.
    """
    # Récupération de tous les points pour le contour
    all_points_flat = []
    for wire in svg_wires:
        for v in wire.vertices():
            all_points_flat.append((v.X, v.Y))
    if not all_points_flat:
        raise ValueError("No SVG paths found with fill to create mold.")
    points_np = np.array(all_points_flat)
    hull = ConvexHull(points_np)
    outline = [points_np[i] for i in hull.vertices]

    # --- Création de la base avec marge ---
    base_outline_wire = cq.Wire.makePolygon([cq.Vector(x, y) for x, y in outline + [outline[0]]])
    # Offset pour la marge
    try:
        base_offset_wires = base_outline_wire.offset2D(margin, "arc")
        if isinstance(base_offset_wires, list) and len(base_offset_wires) > 0:
            base_outline_wire = base_offset_wires[0]
            print(f"Offset de {margin}mm appliqué à l'enveloppe convexe pour la marge.")
        else:
            print(f"Offset de {margin}mm échoué, utilisation de l'enveloppe convexe brute.")
    except Exception as e:
        print(f"Erreur lors de l'offset de l'enveloppe convexe : {e}. Utilisation de l'enveloppe brute.")

    # Diagnostic wire
    print(f"[DEBUG] base_outline_wire type: {type(base_outline_wire)}, isValid: {base_outline_wire.isValid() if hasattr(base_outline_wire, 'isValid') else 'N/A'}, isClosed: {base_outline_wire.IsClosed() if hasattr(base_outline_wire, 'IsClosed') else 'N/A'}")
    if not base_outline_wire.isValid() or not base_outline_wire.IsClosed():
        print("[DEBUG] Wire offset non valide ou non fermé, tentative de reconstruction via makePolygon.")
        pts = [v.toTuple() for v in base_outline_wire.Vertices()]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        base_outline_wire = cq.Wire.makePolygon([cq.Vector(x, y) for x, y in pts])
        print(f"[DEBUG] Wire reconstruit: isValid: {base_outline_wire.isValid()}, isClosed: {base_outline_wire.IsClosed()}")

    # Création de la base
    base_wp = cq.Workplane("XY").add(base_outline_wire).toPending().extrude(base_thickness)
    if not (base_wp.val().isValid() and base_wp.val().ShapeType() == "Solid"):
        raise ValueError("La base du moule (contour SVG) n'est pas un solide valide.")
    base_shape = base_wp.val()

    # --- Génération de la bordure en anneau ---
    try:
        border_outer_wires = base_outline_wire.offset2D(border_thickness, "arc")
        if isinstance(border_outer_wires, list) and len(border_outer_wires) > 0:
            border_outer_wire = border_outer_wires[0]
            print(f"Offset extérieur de {border_thickness}mm appliqué pour la bordure.")
        else:
            print(f"Offset extérieur échoué, pas de bordure.")
            border_outer_wire = None
    except Exception as e:
        print(f"Erreur lors de l'offset extérieur pour la bordure : {e}. Pas de bordure.")
        border_outer_wire = None

    border_inner_wire = base_outline_wire  # L'intérieur de la bordure est la base (avec marge)

    if border_outer_wire is not None:
        # Solide extérieur
        border_outer_wp = cq.Workplane("XY").add(border_outer_wire).toPending().extrude(base_thickness + border_height)
        border_outer_shape = border_outer_wp.val()
        # Solide intérieur (même hauteur)
        border_inner_wp = cq.Workplane("XY").add(border_inner_wire).toPending().extrude(base_thickness + border_height)
        border_inner_shape = border_inner_wp.val()
        # Anneau = extérieur - intérieur
        border_ring = border_outer_shape.cut(border_inner_shape)
        # Fusionne base et anneau
        mold = base_shape.fuse(border_ring)
        print("Bordure en anneau générée et fusionnée à la base.")
    else:
        mold = base_shape
        print("Pas de bordure générée, seule la base est utilisée.")

    # --- Export STL de la base avant gravure dans debug_stl/ ---
    base_stl_path = os.path.join(debug_dir, os.path.basename(base_stl_name))
    if export_base_stl:
        cq.exporters.export(mold, base_stl_path)
        print(f"STL de la base (avec bordure) exporté : {base_stl_path}")
    return mold, base_outline_wire

def engrave_polygons(mold, svg_wires, shape_history, base_thickness, engrave_depth, export_steps, debug_dir):
    """
    Grave les polygones sur le moule, met à jour shape_history['engraved'] pour chaque shape.
    Retourne le moule gravé et la liste des indices gravés.
    """
    engraved_indices = []
    # Utiliser le regroupement par path_idx puis inclusion
    grouped_wires = group_wires_by_inclusion(svg_wires, shape_history)
    wire_to_shape = shape_history.get('wire_to_shape', {})
    for idx, wire_group in enumerate(grouped_wires):
        try:
            print(f"\n--- Gravure polygone {idx} (groupe de {len(wire_group)} wires) ---")
            for w_idx, w in enumerate(wire_group):
                try:
                    global_wire_index = svg_wires.index(w)
                except ValueError:
                    global_wire_index = None
                shape_info = wire_to_shape.get(global_wire_index, None)
                if shape_info is not None:
                    hist = shape_history[shape_info]
                    print(f"  Wire {w_idx} (global idx {global_wire_index}): SVG path_idx={hist['svg_path_idx']}, sub_idx={hist['svg_sub_idx']}, nb_points={len(hist['simplified_points'])}")
                else:
                    print(f"  Wire {w_idx} (global idx {global_wire_index}): [shape SVG non trouvée]")
                print(f"    isClosed: {w.IsClosed()}, isValid: {w.isValid()}, nb sommets: {len(list(w.Vertices()))}")
            try:
                face_result = cq.Face.makeFromWires(wire_group[0], wire_group[1:])
            except Exception as e:
                print(f"Erreur makeFromWires sur groupe {idx}: {e}")
                for w in wire_group:
                    try:
                        global_wire_index = svg_wires.index(w)
                    except ValueError:
                        global_wire_index = None
                    if global_wire_index is not None:
                        for shape_key, hist in shape_history.items():
                            if isinstance(shape_key, tuple) and hist.get('cq_wire_index') == global_wire_index:
                                hist['engraved'] = False
                    else:
                        print(f"Impossible de trouver le wire: {w} dans svg_wires pour le groupe {idx}.")
                continue
            if isinstance(face_result, list):
                if not face_result:
                    print(f"Polygone {idx} : makeFromWires a retourné une liste vide.")
                    continue
                face = face_result[0]
            else:
                face = face_result
            engraving = cq.Workplane("XY").add(face).extrude(-engrave_depth)
            engravings = flatten_cq_solids(engraving)
            if not engravings:
                print(f"Warning: Extrusion du groupe {idx} n'a pas produit de solide.")
                for w in wire_group:
                    try:
                        global_wire_index = svg_wires.index(w)
                    except ValueError:
                        global_wire_index = None
                    if global_wire_index is not None:
                        for shape_key, hist in shape_history.items():
                            if isinstance(shape_key, tuple) and hist.get('cq_wire_index') == global_wire_index:
                                hist['engraved'] = False
                    else:
                        print(f"Impossible de trouver le wire: {w} dans svg_wires pour le groupe {idx}.")
                continue
            success = False
            for engraving_solid in engravings:
                if engraving_solid.isValid() and engraving_solid.ShapeType() == "Solid":
                    engraving_solid = engraving_solid.translate((0, 0, base_thickness))
                    try:
                        new_mold = mold.cut(engraving_solid)
                        mold = new_mold
                        success = True
                        if export_steps:
                            step_stl_path = os.path.join(debug_dir, f"step_{idx}.stl")
                            cq.exporters.export(new_mold, step_stl_path)
                            print(f"Export STL intermédiaire réussi : {step_stl_path}")
                    except Exception as ce:
                        print(f"Erreur lors du cut avec l'extrusion du groupe {idx} : {ce}")
                else:
                    print(f"Warning: Extrusion du groupe {idx} n'a pas produit un solide valide.")
            if success:
                engraved_indices.append(idx)
            # Mise à jour de shape_history pour indiquer que le groupe a été gravé
            print(f"Groupe {idx} gravé avec succès : {success}.")
            for w in wire_group:
                try:
                    global_wire_index = svg_wires.index(w)
                    for shape_key, hist in shape_history.items():
                        if isinstance(shape_key, tuple) and hist.get('cq_wire_index') == global_wire_index:
                            shape_history[shape_key]['engraved'] = success
                except Exception:
                    pass
        except Exception as e:
            print(f"Erreur lors de la gravure du groupe {idx}: {e}")
    return mold, engraved_indices

def generate_cadquery_mold(svg_file, max_dim, base_thickness, border_height, border_thickness, engrave_depth, margin=MARGE, export_base_stl=True, base_stl_name="moule_base.stl", export_steps=False):
    normalized_svg_file = normalize_svg_fill(svg_file)
    svg_wires, shape_history = svg_to_cadquery_wires(normalized_svg_file, max_dim)

    # Initialisation du statut 'engraved' pour chaque shape
    for k in shape_history:
        if isinstance(k, tuple):
            shape_history[k]['engraved'] = False

    debug_dir = "debug_stl"
    os.makedirs(debug_dir, exist_ok=True)

    mold, base_outline_wire = create_mold_base(
        svg_wires, margin, base_thickness, border_height, border_thickness, debug_dir, base_stl_name, export_base_stl=export_base_stl
    )

    mold, engraved_indices = engrave_polygons(
        mold, svg_wires, shape_history, base_thickness, engrave_depth, export_steps, debug_dir
    )

    return mold, engraved_indices, shape_history

def generate_summary_svg(original_svg_path, shape_keys, output_svg_name, shape_history=None):
    """
    Génère un SVG de résumé en dessinant explicitement chaque shape (polygone) à partir des points simplifiés de shape_history,
    coloré en noir si engraved, rouge sinon.
    """
    tree = ET.parse(original_svg_path)
    root = tree.getroot()

    # Suppression des éléments non désirés
    for bad_tag in ['metadata', 'desc', 'title']:
        for elem in root.findall(f'.//{bad_tag}'):
            parent = find_parent(root, elem)
            if parent is not None:
                parent.remove(elem)

    # Création d'un groupe pour les polygones sélectionnés
    group = ET.Element('g', {'id': 'summary_polygons'})
    root.append(group)

    # Ajout des polygones colorés à partir des points simplifiés
    if shape_history is not None:
        for (path_idx, sub_idx) in shape_keys:
            hist = shape_history.get((path_idx, sub_idx), None)
            if hist is not None:
                pts = hist.get('simplified_points', [])
                if len(pts) >= 3:
                    points_str = ' '.join(f"{x},{y}" for x, y in pts)
                    engraved = hist.get('engraved', False)
                    style = 'fill:black;stroke:black;stroke-width:1' if engraved else 'fill:red;stroke:black;stroke-width:1'
                    polygon_elem = ET.Element('polygon', {
                        'points': points_str,
                        'style': style,
                        'data-pathidx': str(path_idx),
                        'data-subidx': str(sub_idx)
                    })
                    group.append(polygon_elem)

        # Écriture du nouveau fichier SVG
        tree.write(output_svg_name, encoding='utf-8', xml_declaration=True)
        print(f"SVG généré : {output_svg_name}")
    else:
        print("Aucune shape_history fournie pour générer le SVG de résumé.")


def flatten_cq_solids(obj):
    """Aplati récursivement une structure de listes, Workplane, Shape pour ne garder que les objets CadQuery valides (ayant .wrapped)."""
    solids = []
    if isinstance(obj, (list, tuple)):
        for item in obj:
            solids.extend(flatten_cq_solids(item))
    elif hasattr(obj, 'wrapped'):
        solids.append(obj)
    elif hasattr(obj, 'val') and callable(obj.val):
        # Workplane avec un seul objet
        solids.extend(flatten_cq_solids(obj.val()))
    elif hasattr(obj, 'vals') and callable(obj.vals):
        # Workplane avec plusieurs objets
        for v in obj.vals():
            solids.extend(flatten_cq_solids(v))
    else:
        print(f"[flatten_cq_solids] Objet ignoré (ni CQ, ni Workplane, ni liste): {type(obj)} -> {obj}")
    return solids

def group_wires_by_inclusion(wires, shape_history=None):
    """
    Regroupe les wires CadQuery pour extrusion, en tenant compte de leur origine SVG.

    - Si shape_history est fourni et contient 'wire_to_shape',
      les wires sont d'abord regroupés par svg_path_idx (c'est-à-dire par élément <path> du SVG).
      À l'intérieur de chaque groupe, on applique la logique d'inclusion géométrique (trous/contours).
    - Si shape_history n'est pas fourni, on applique la logique d'inclusion sur tous les wires.

    Args:
        wires: liste de wires CadQuery
        shape_history: dict optionnel, issu de svg_to_cadquery_wires
    Returns:
        Liste de groupes de wires, chaque groupe étant [outer, inner1, ...]
    """
    if shape_history is None or 'wire_to_shape' not in shape_history:
        # Fallback : ancienne logique sur tous les wires
        grouped = []
        wires_polygons = []
        # Conversion de chaque wire en polygone Shapely
        for w in wires:
            pts = [(v.X, v.Y) for v in w.Vertices()]
            wires_polygons.append((w, Polygon(pts)))
        used = set()
        for i, (outer_w, outer_poly) in enumerate(wires_polygons):
            if i in used:
                continue  # Ce wire a déjà été utilisé comme inner
            inners = []
            # Recherche de tous les wires strictement inclus dans outer_poly
            for j, (inner_w, inner_poly) in enumerate(wires_polygons):
                if i != j and outer_poly.contains(inner_poly):
                    inners.append(inner_w)
                    used.add(j)
            grouped.append([outer_w] + inners)
            used.add(i)
        return grouped

    # 1. Regrouper les indices de wires par svg_path_idx
    wire_to_shape = shape_history['wire_to_shape']
    path_idx_to_wire_indices = {}
    for wire_idx, shape_key in wire_to_shape.items():
        svg_path_idx = shape_history[shape_key]['svg_path_idx']
        # On regroupe tous les indices de wires ayant le même svg_path_idx
        path_idx_to_wire_indices.setdefault(svg_path_idx, []).append(wire_idx)

    all_groups = []
    # Pour chaque groupe issu du même <path> SVG
    for path_idx, wire_indices in path_idx_to_wire_indices.items():
        # Récupérer les wires de ce path_idx
        wires_in_group = [wires[i] for i in wire_indices]
        # Conversion en polygones Shapely pour inclusion rapide
        wires_polygons = []
        for w in wires_in_group:
            pts = [(v.X, v.Y) for v in w.Vertices()]
            wires_polygons.append((w, Polygon(pts)))
        used = set()
        # Logique d'inclusion (contour principal + trous) dans ce groupe
        for i, (outer_w, outer_poly) in enumerate(wires_polygons):
            if i in used:
                continue  # Ce wire a déjà été utilisé comme inner
            inners = []
            for j, (inner_w, inner_poly) in enumerate(wires_polygons):
                if i != j and outer_poly.contains(inner_poly):
                    inners.append(inner_w)
                    used.add(j)
            # Ajoute le groupe [outer, inner1, ...] à la liste globale
            all_groups.append([outer_w] + inners)
            used.add(i)
    return all_groups

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère un moule à silicone depuis un SVG avec CadQuery.")
    parser.add_argument("--svg", help="Chemin du fichier SVG")
    parser.add_argument("--size", type=float, default=MAX_DIMENSION, help="Taille max du moule (mm)")
    parser.add_argument("--output", default="moule_cadquery.stl", help="Fichier de sortie STL")
    parser.add_argument("--no-interactive", action="store_true", help="Désactive le mode interactif")
    parser.add_argument("--export-steps", action="store_true", help="Exporter les étapes intermédiaires en STL")
    args = parser.parse_args()

    if not os.path.exists(args.svg):
        print(f"Le fichier SVG {args.svg} n'existe pas. Création d'un fichier SVG de test.")
        with open("test_rect.svg", "w") as f:
            f.write("<svg width=\"100\" height=\"100\" xmlns=\"http://www.w3.org/2000/svg\"><rect x=\"10\" y=\"10\" width=\"80\" height=\"80\" fill=\"blue\" /></svg>")
        args.svg = "test_rect.svg"

    ignored_indices = []
    try:
        mold, engraved_indices, shape_history = generate_cadquery_mold(
            args.svg,
            args.size,
            BASE_THICKNESS,
            BORDER_HEIGHT,
            BORDER_THICKNESS,
            ENGRAVE_DEPTH,
            margin=MARGE,
            export_base_stl=True,
            base_stl_name="moule_base.stl",
            export_steps=args.export_steps if hasattr(args, 'export_steps') else False
        )
        # Export final
        cq.exporters.export(mold, args.output)
        print(f"Moule CadQuery généré : {args.output}")
        engraved_shape_keys = [k for k, v in shape_history.items() if isinstance(k, tuple) and v.get('engraved')]
        ignored_shape_keys = [k for k, v in shape_history.items() if isinstance(k, tuple) and not v.get('engraved')]
        # Génération du SVG de résumé avec couleurs distinctes
        all_shape_keys = [k for k in shape_history if isinstance(k, tuple)]
        if all_shape_keys:
            generate_summary_svg(args.svg, all_shape_keys, 'summary_polygons.svg', shape_history=shape_history)
            print("SVG de résumé généré avec couleurs selon le statut de gravure.")
        else:
            print("Aucun polygone détecté, pas de SVG de visualisation généré.")
    except ValueError as e:
        print(f"Erreur lors de la génération du moule : {e}")


