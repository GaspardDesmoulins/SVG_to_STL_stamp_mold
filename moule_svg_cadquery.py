import cadquery as cq
from svgpathtools import svg2paths2
import numpy as np
from pathlib import Path
import os
import shutil
from xml.etree import ElementTree as ET
import re
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon as ShapelyPolygon
import math
from settings import BASE_THICKNESS, BORDER_HEIGHT, BORDER_THICKNESS, \
    MARGE, ENGRAVE_DEPTH, MAX_DIMENSION

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
    # Détermine le chemin du fichier normalisé
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(svg_file_path))[0]
        normd_file_path = os.path.join(debug_dir, f"{base_name}_normalized.svg")
        with open(normd_file_path, "w") as f:
            f.write(normalized_svg)
    else:
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


def svg_to_cadquery_wires(svg_file, max_dimension=MAX_DIMENSION, interactive=True, force_all_contours=False):
    """
    Convertit un SVG en wires CadQuery, avec simplification et gestion de l'échelle.
    Peut forcer l'extraction de tous les contours (utile pour les SVG Affinity Designer).
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

    def is_visible(attr):
        # Détermine si un élément doit être extrait (fill ou stroke visible, ou extraction forcée)
        if force_all_contours:
            return True
        fill = attr.get('fill', None)
        stroke = attr.get('stroke', None)
        # On considère visible si fill n'est pas none/transparent OU stroke n'est pas none/transparent
        def is_none_or_transparent(val):
            if val is None:
                return False  # Si absent, on considère visible
            val = val.strip().lower()
            return val in ['none', 'transparent']
        return not (is_none_or_transparent(fill) and is_none_or_transparent(stroke))

    for path_idx, (path, attr) in enumerate(zip(paths, attributes)):
        if not is_visible(attr):
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

    # --- Ajout du support des ellipses SVG ---
    tree = ET.parse(str(svgfile.absolute()))
    root = tree.getroot()
    ellipse_count = 0
    for elem in root.iter():
        if strip_namespace(elem.tag) == 'ellipse':
            try:
                # On récupère les attributs SVG (fill, stroke, etc.)
                fill = elem.attrib.get('fill', None)
                stroke = elem.attrib.get('stroke', None)
                if not force_all_contours:
                    def is_none_or_transparent(val):
                        if val is None:
                            return False
                        val = val.strip().lower()
                        return val in ['none', 'transparent']
                    if is_none_or_transparent(fill) and is_none_or_transparent(stroke):
                        continue  # Ellipse non visible
                cx = float(elem.attrib.get('cx', '0'))
                cy = float(elem.attrib.get('cy', '0'))
                rx = float(elem.attrib.get('rx', '0'))
                ry = float(elem.attrib.get('ry', '0'))
                if rx == 0 or ry == 0:
                    continue  # Ellipse dégénérée
                n_pts = 60
                ellipse_points = []
                for i in range(n_pts):
                    theta = 2 * math.pi * i / n_pts
                    x = cx + rx * math.cos(theta)
                    y = cy + ry * math.sin(theta)
                    ellipse_points.append([x, y])
                ellipse_points.append(ellipse_points[0])  # Fermeture
                all_points.extend(ellipse_points)
                shapes_data.append(ellipse_points)
                ellipse_key = ("ellipse", ellipse_count)
                shape_keys.append(ellipse_key)
                shape_history[ellipse_key] = {
                    'svg_path_idx': "ellipse",
                    'svg_sub_idx': ellipse_count,
                    'svg_attr': dict(elem.attrib),
                    'sampled_points': ellipse_points,
                    'simplified_points': ellipse_points,
                    'cq_wire_index': None,
                }
                ellipse_count += 1
            except Exception as e:
                print(f"Erreur lors de l'extraction d'une ellipse : {e}")

    if not all_points:
        raise ValueError("No SVG paths found with fill or stroke to create mold.")

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
        base_mold = base_shape.fuse(border_ring)
        print("Bordure en anneau générée et fusionnée à la base.")
    else:
        base_mold = base_shape
        print("Pas de bordure générée, seule la base est utilisée.")

    # --- Export STL de la base avant gravure dans debug_stl/ ---
    base_stl_path = os.path.join(debug_dir, os.path.basename(base_stl_name))
    if export_base_stl:
        cq.exporters.export(base_mold, base_stl_path)
        print(f"STL de la base (avec bordure) exporté : {base_stl_path}")
    return base_mold

def engrave_polygons(mold, svg_wires, shape_history, base_thickness, engrave_depth, export_steps, debug_dir,
                      original_svg_path=None, shape_keys=None):
    """
    Grave les polygones sur le moule, met à jour shape_history['engraved'] pour chaque shape.
    Retourne le moule gravé et la liste des indices gravés.
    Génère un SVG de résumé à chaque étape de gravure.
    """
    engraved_indices = []
    # Utiliser le regroupement par path_idx puis inclusion
    grouped_wires = group_wires_by_inclusion(svg_wires, shape_history)
    wire_to_shape = shape_history.get('wire_to_shape', {})
    for group_idx, wire_group in grouped_wires:
        try:
            print(f"\n--- Gravure polygone {group_idx} (groupe de {len(wire_group)} wires) ---")
            # on construit un face à partir des wires du groupe
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
                print(f"Erreur makeFromWires sur groupe {group_idx}: {e}")
                mark_wires_not_engraved(wire_group, svg_wires, shape_history, group_idx)
                continue
            if isinstance(face_result, list):
                if not face_result:
                    print(f"Polygone {group_idx} : makeFromWires a retourné une liste vide.")
                    continue
                face = face_result[0]
            else:
                face = face_result
            
            #on tente d'extruder la face en extrusion directe
            engraving = cq.Workplane("XY").add(face).extrude(-engrave_depth)
            print(f"[DEBUG] Type engraving après extrusion directe: {type(engraving)}")

            # si l'extrusion directe échoue, on utilise la dépouille
            if engraving is None:
                # --- Gravure avec dépouille (loft) ---
                draft_angle_deg = 15  # Angle de dépouille en degrés (modifiable)
                engraving = loft_with_draft(wire_group, draft_angle_deg, engrave_depth)
                print(f"[DEBUG] Type engraving retourné par loft_with_draft: {type(engraving)}")

            # Simplification des solides extrudés
            engravings = flatten_cq_solids(engraving)
            print(f"[DEBUG] Types des objets retournés par flatten_cq_solids: {[type(e) for e in engravings]}")
            if not engravings:
                print(f"Warning: Extrusion du groupe {group_idx} n'a pas produit de solide.")
                mark_wires_not_engraved(wire_group, svg_wires, shape_history, group_idx)
                continue
            success = False
            for engraving_solid in engravings:
                print(f"[DEBUG] engraving_solid type: {type(engraving_solid)}")
                # Vérifie la présence de isValid()
                if hasattr(engraving_solid, 'isValid'):
                    valid = engraving_solid.isValid()
                else:
                    valid = False
                print(f"[DEBUG] engraving_solid.isValid() = {valid}")
                if valid and hasattr(engraving_solid, 'ShapeType') and engraving_solid.ShapeType() in ["Solid", "Compound"]:
                    engraving_solid = engraving_solid.translate((0, 0, base_thickness))
                    try:
                        # Gravure explicite du moule par appel de cut()
                        new_mold = mold.cut(engraving_solid)
                        mold = new_mold
                        success = True
                        if export_steps:
                            step_stl_path = os.path.join(debug_dir, f"step_{group_idx}.stl")
                            cq.exporters.export(new_mold, step_stl_path)
                            print(f"Export STL intermédiaire réussi : {step_stl_path}")
                        # --- Génération du SVG de résumé pour cette étape ---
                        if original_svg_path is not None and shape_keys is not None:
                            summary_svg_path = os.path.join(debug_dir, f"step_{group_idx}_summary.svg")
                            generate_summary_svg(original_svg_path, shape_keys, summary_svg_path, shape_history)
                    except Exception as ce:
                        print(f"Erreur lors du cut avec l'extrusion du groupe {group_idx} : {ce}")
                else:
                    print(f"Warning: Extrusion du groupe {group_idx} n'a pas produit un solide valide.")
            if success:
                engraved_indices.append(group_idx)
            # Mise à jour de shape_history pour indiquer que le groupe a été gravé
            print(f"Groupe {group_idx} gravé avec succès : {success}.")
            mark_wires_not_engraved(wire_group, svg_wires, shape_history, group_idx)
        except Exception as e:
            print(f"Erreur lors de la gravure du groupe {group_idx}: {e}")
    return mold, engraved_indices

def generate_cadquery_mold(svg_file, max_dim, base_thickness=BASE_THICKNESS, border_height=BORDER_HEIGHT,
                           border_thickness=BORDER_THICKNESS, engrave_depth=ENGRAVE_DEPTH, margin=MARGE,
                           export_base_stl=True, base_stl_name="moule_base.stl", export_steps=False,
                           keep_debug_files=False):
    # Détermination du nom du dossier de debug à partir du nom du fichier SVG
    svg_basename = os.path.splitext(os.path.basename(svg_file))[0]
    debug_dir = f"debug_{svg_basename}"
    os.makedirs(debug_dir, exist_ok=True)

    normalized_svg_file = normalize_svg_fill(svg_file, debug_dir=debug_dir)
    svg_wires, shape_history = svg_to_cadquery_wires(normalized_svg_file, max_dim)

    # Initialisation du statut 'engraved' pour chaque shape
    for k in shape_history:
        if isinstance(k, tuple):
            shape_history[k]['engraved'] = False

    mold = create_mold_base(
        svg_wires, margin, base_thickness, border_height, border_thickness, debug_dir, base_stl_name, export_base_stl=export_base_stl
    )

    # Récupération de la liste des shape_keys pour le résumé SVG
    shape_keys = [k for k in shape_history if isinstance(k, tuple)]

    mold, engraved_indices = engrave_polygons(
        mold, svg_wires, shape_history, base_thickness, engrave_depth, export_steps, debug_dir,
        original_svg_path=svg_file, shape_keys=shape_keys
    )

    # Suppression du dossier de debug si demandé
    if not keep_debug_files:
        shutil.rmtree(debug_dir, ignore_errors=True)

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
        val = obj.val()
        print(f"[flatten_cq_solids] Workplane.val() type: {type(val)}")
        solids.extend(flatten_cq_solids(val))
    elif hasattr(obj, 'vals') and callable(obj.vals):
        # Workplane avec plusieurs objets
        for v in obj.vals():
            print(f"[flatten_cq_solids] Workplane.vals() type: {type(v)}")
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
        for wire in wires:
            # Conversion de chaque wire en polygone Shapely
            points = [(vertex.X, vertex.Y) for vertex in wire.Vertices()]
            wires_polygons.append((wire, ShapelyPolygon(points)))
        used_indices = set()
        while len(used_indices) < len(wires_polygons):
            # Recherche du wire qui contient le plus d'autres wires non utilisés
            max_contained = -1
            outer_index = None
            for candidate_index, (candidate_wire, candidate_poly) in enumerate(wires_polygons):
                if candidate_index in used_indices:
                    continue
                contained_indices = []
                for test_index, (test_wire, test_poly) in enumerate(wires_polygons):
                    if candidate_index != test_index and test_index not in used_indices and candidate_poly.contains(test_poly):
                        contained_indices.append(test_index)
                if len(contained_indices) > max_contained:
                    max_contained = len(contained_indices)
                    outer_index = candidate_index
                    inners_indices = contained_indices
            if outer_index is not None:
                # On forme le groupe avec l'outer et ses inners
                group = [wires_polygons[outer_index][0]] + [wires_polygons[j][0] for j in inners_indices]
                grouped.append((outer_index, group))
                used_indices.add(outer_index)
                used_indices.update(inners_indices)
            else:
                # Aucun wire restant ne contient d'autres, on ajoute les restants seuls
                for idx in range(len(wires_polygons)):
                    if idx not in used_indices:
                        grouped.append((idx, [wires_polygons[idx][0]]))
                        used_indices.add(idx)
        return grouped

    # 1. Regrouper les indices de wires par svg_path_idx
    wire_to_shape = shape_history['wire_to_shape']
    path_idx_to_wire_indices = {}
    for wire_index, shape_key in wire_to_shape.items():
        svg_path_idx = shape_history[shape_key]['svg_path_idx']
        path_idx_to_wire_indices.setdefault(svg_path_idx, []).append(wire_index)

    all_groups = []
    # Pour chaque groupe issu du même <path> SVG
    for path_idx, wire_indices in path_idx_to_wire_indices.items():
        # Récupérer les wires de ce path_idx
        wires_in_group = [wires[i] for i in wire_indices]
        wires_polygons = []
        for wire in wires_in_group:
            points = [(vertex.X, vertex.Y) for vertex in wire.Vertices()]
            wires_polygons.append((wire, ShapelyPolygon(points)))
        used_indices = set()
        while len(used_indices) < len(wires_polygons):
            # Recherche du wire qui contient le plus d'autres wires non utilisés
            max_contained = -1
            outer_index = None
            for candidate_index, (candidate_wire, candidate_poly) in enumerate(wires_polygons):
                if candidate_index in used_indices:
                    continue
                contained_indices = []
                for test_index, (test_wire, test_poly) in enumerate(wires_polygons):
                    if candidate_index != test_index and test_index not in used_indices and candidate_poly.contains(test_poly):
                        contained_indices.append(test_index)
                if len(contained_indices) > max_contained:
                    max_contained = len(contained_indices)
                    outer_index = candidate_index
                    inners_indices = contained_indices
            if outer_index is not None:
                # On forme le groupe avec l'outer et ses inners
                group = [wires_polygons[outer_index][0]] + [wires_polygons[j][0] for j in inners_indices]
                all_groups.append((path_idx, group))
                used_indices.add(outer_index)
                used_indices.update(inners_indices)
                print(f"  Group formed. Outer wire: {wires_polygons[outer_index][0]}, inners: {[wires_polygons[j][0] for j in inners_indices]}")
            else:
                # Aucun wire restant ne contient d'autres, on ajoute les restants seuls
                for idx in range(len(wires_polygons)):
                    if idx not in used_indices:
                        all_groups.append((path_idx, [wires_polygons[idx][0]]))
                        used_indices.add(idx)
                        print(f"  Group formed. Single wire: {wires_polygons[idx][0]} (no inners)")
    # Vérification que tous les path_idx ont été utilisés
    used_path_indices = {g[0] for g in all_groups}
    for path_idx in path_idx_to_wire_indices.keys():
        if path_idx not in used_path_indices:
            print(f"[WARNING] Path index {path_idx} not used in any group, possible issue with SVG structure.")
    return all_groups

def scale_wire_2d(wire, scale_factor):
    """Retourne une nouvelle wire dilatée/réduite par rapport à son centroïde."""
    pts = np.array([(v.X, v.Y) for v in wire.Vertices()])
    centroid = pts.mean(axis=0)
    scaled_pts = centroid + (pts - centroid) * scale_factor
    return cq.Wire.makePolygon([cq.Vector(x, y) for x, y in scaled_pts])

def offset_polygon_along_normals(points, offset_distance, centroid_direction=1):
    """
    Décale chaque point d'un polygone selon la normale locale à la bordure.
    - points : liste de tuples (x, y) du polygone (doit être fermé ou sera fermé automatiquement)
    - offset_distance : distance de décalage (positive)
    - centroid_direction : +1 pour outer (vers l'extérieur du centroïde), -1 pour inner (vers le centroïde)
    Retourne une nouvelle liste de points décalés.
    """
    # Filtre les points dupliqués consécutifs
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
    centroid = pts.mean(axis=0)
    offset_points = []
    for i in range(num_points):
        prev_point = closed_pts[(i - 1) % num_points]
        current_point = closed_pts[i]
        next_point = closed_pts[(i + 1) % num_points]
        # Vecteurs des segments adjacents
        vec_prev = current_point - prev_point
        vec_next = next_point - current_point
        # Normales aux segments (sens trigo)
        normal_prev = np.array([-vec_prev[1], vec_prev[0]])
        normal_next = np.array([-vec_next[1], vec_next[0]])
        norm_prev = np.linalg.norm(normal_prev)
        norm_next = np.linalg.norm(normal_next)
        if norm_prev < 1e-10 or norm_next < 1e-10:
            # Si un segment est nul, saute la normale correspondante
            normal = normal_prev if norm_next < 1e-10 else normal_next
            if np.linalg.norm(normal) < 1e-10:
                normal = np.array([1.0, 0.0])  # Valeur par défaut
        else:
            normal_prev /= norm_prev
            normal_next /= norm_next
            # Normale moyenne au sommet
            normal = (normal_prev + normal_next)
            if np.linalg.norm(normal) < 1e-8:
                normal = normal_prev  # Cas angle plat
            else:
                normal /= np.linalg.norm(normal)
        # Sens de la normale (vers l'extérieur ou l'intérieur)
        to_centroid = centroid - current_point
        if np.dot(normal, to_centroid) * centroid_direction > 0:
            normal = -normal
        offset_point = current_point + offset_distance * normal
        offset_points.append(tuple(offset_point))
    # Ferme le polygone
    offset_points.append(offset_points[0])
    if not ShapelyPolygon(offset_points).is_simple:
        print("[WARNING] Polygone offset non simple, possible auto-intersection")
    return offset_points

def get_scaled_wire_from_wire(wire, offset, centroid_direction):
    """
    Génère un wire décalé (scaled) à partir d'un wire d'origine, selon la normale locale et un offset.
    Args:
        wire (cq.Wire): Wire d'origine
        offset (float): Distance de décalage
        direction_centroid (int): +1 pour outer, -1 pour inner
    Returns:
        cq.Wire: Wire décalé
    """
    wire_points = [(vertex.X, vertex.Y) for vertex in wire.Vertices()]
    offset_points = offset_polygon_along_normals(wire_points, offset, centroid_direction=centroid_direction)
    return cq.Wire.makePolygon([cq.Vector(x, y) for x, y in offset_points])


def loft_with_draft(wire_group, draft_angle_deg, depth):
    """
    Réalise un loft avec dépouille pour chaque wire du groupe (contour principal et éventuels trous).
    Pour chaque wire (contour) :
      - Génère un wire décalé selon la normale locale (dépouille)
      - Fait un loft entre le wire original (en Z bas) et le wire décalé (en Z haut)
      - Le solide du contour principal (outer) est ajouté, les solides des trous (inners) sont soustraits (cut)
    Args:
        wire_group (list): Liste de wires CadQuery (le premier est le contour principal, les suivants sont les trous)
        draft_angle_deg (float): Angle de dépouille en degrés (positif = évasé)
        depth (float): Profondeur du motif (positive)
    Returns:
        Solid CadQuery résultant du loft avec dépouille, ou None si échec
    """
    # Si pas de dépouille ou profondeur nulle, rien à faire
    if depth == 0 or draft_angle_deg == 0:
        return None  # Pas de dépouille à appliquer
    try:
        # Calcul de l'offset latéral à appliquer selon l'angle de dépouille
        offset = abs(depth) * math.tan(math.radians(draft_angle_deg))
        print("[loft_with_draft] Début du loft individuel. Nombre de wires:", len(wire_group))
        lofted_solids = []
        # Parcours de chaque wire du groupe (contour principal + trous)
        for wire_index, wire in enumerate(wire_group):
            # Sens de l'offset : +1 pour outer, -1 pour inners
            centroid_direction = 1 if wire_index == 0 else -1
            # Génération du wire décalé via la fonction dédiée
            scaled_wire = get_scaled_wire_from_wire(wire, offset, centroid_direction)
            # Création des faces (top = wire décalé, bottom = wire original)
            try:
                face_top = cq.Face.makeFromWires(scaled_wire)
                face_bottom = cq.Face.makeFromWires(wire)
            except Exception as e:
                print(f"[loft_with_draft] Echec de makeFromWires (individuel) : {e}")
                continue  # Passe au wire suivant si échec
            # Placement des faces à la bonne hauteur Z
            face_top = face_top.translate((0, 0, 0))
            face_bottom = face_bottom.translate((0, 0, -depth))
            # Création du loft entre les deux faces
            workplane = cq.Workplane("XY").add([face_top, face_bottom])
            try:
                lofted_solid = workplane.loft(combine=True)
                print(f"[loft_with_draft] Loft individuel réussi pour wire {wire_index}.")
                lofted_solids.append(lofted_solid)
            except Exception as e:
                print(f"[loft_with_draft] Echec du loft individuel pour wire {wire_index}: {e}")
                continue  # Passe au wire suivant si échec
        # Combine les solides : outer (index 0) en add, inners (>0) en cut
        if not lofted_solids:
            print("[loft_with_draft] Aucun solide généré.")
            return None
        # Le premier solide (contour principal) sert de base
        result_solid = lofted_solids[0]
        # Les suivants (trous) sont soustraits
        for inner_solid in lofted_solids[1:]:
            try:
                result_solid = result_solid.cut(inner_solid)
            except Exception as e:
                print(f"[loft_with_draft] Echec du cut pour inner : {e}")
        return result_solid
    except Exception as e:
        print(f"[loft_with_draft] Echec du loft avec dépouille (individuel) : {e}")
        return None

def mark_wires_not_engraved(wire_group, svg_wires, shape_history, group_idx):
    """
    Marque les wires du groupe comme non gravés dans shape_history.
    """
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
            print(f"Impossible de trouver le wire: {w} dans svg_wires pour le groupe {group_idx}.")

