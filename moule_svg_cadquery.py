import os
import shutil
import logging
import math
import cadquery as cq
import numpy as np
from svgpathtools import svg2paths2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
from xml.etree import ElementTree as ET
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
from settings import BASE_THICKNESS, BORDER_HEIGHT, BORDER_THICKNESS, \
    MARGE, ENGRAVE_DEPTH, MAX_DIMENSION
from utils import rdp, normalize_svg_fill, extract_subpaths, \
    strip_namespace, filter_points, resample_polygon, align_resampled_to_reference

# --- Logger configuration ---
logger = logging.getLogger("moule_svg_cadquery")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def svg_to_cadquery_wires(svg_file, max_dimension=MAX_DIMENSION, interactive=True, force_all_contours=False):
    """
    Convertit un SVG en wires CadQuery, avec simplification et gestion de l'échelle.
    Peut forcer l'extraction de tous les contours (utile pour les SVG Affinity Designer).
    Retourne la liste des wires et l'historique des shapes.
    """
    logger.info("Début extraction des paths SVG...")
    svgfile = Path(svg_file)
    paths, attributes, svg_attr = svg2paths2(str(svgfile.absolute()))
    all_points = []
    shapes_data = []
    ignored_indices = []
    sampling_default = 30
    shape_history = {}
    shape_keys = []
    wire_to_shape = {}

    # --- Extraction simple des éléments <path> (le SVG est déjà aplati, donc pas de transformation à appliquer) ---
    tree = ET.parse(str(svgfile.absolute()))
    root = tree.getroot()
    svg_path_elements = [el for el in root.iter() if strip_namespace(el.tag) == 'path']
    # On associe chaque élément <path> à la matrice identité (aucune transformation restante)
    path_with_transform = [(path_elem, np.eye(3)) for path_elem in svg_path_elements]
    # --- Fin extraction simplifiée ---

    def is_visible(attr):
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

    # --- Étape 2 : Application des transformations à chaque path SVG ---
    # On parcourt les paths extraits par svg2paths2, mais on applique la matrice trouvée à chaque point du path.
    # On suppose que l'ordre des paths svgpathtools et des éléments XML <path> est identique (plus robuste que la comparaison d'attributs).
    for path_idx, (svgpathtools_path, svgpathtools_attr) in enumerate(zip(paths, attributes)):
        if not is_visible(svgpathtools_attr):
            continue
        logger.debug(f"--- Path {path_idx} ---")
        # Récupère la matrice de transformation cumulée pour ce path
        if path_idx < len(path_with_transform):
            cumulative_transform = path_with_transform[path_idx][1]
        else:
            cumulative_transform = np.eye(3)
        # Extraction des sous-chemins (subpaths) du path SVG
        subpaths = extract_subpaths(svgpathtools_path, sampling_default)
        logger.debug(f"Subpaths {len(subpaths)}")
        for sub_idx, sampled_points in enumerate(subpaths):
            if sampled_points:
                # --- Application de la matrice de transformation à chaque point du path ---
                # On transforme chaque point [x, y] en [x', y'] via la matrice SVG cumulée
                transformed_points = []
                for pt in sampled_points:
                    # On homogénéise le point pour la matrice 3x3
                    pt_homogeneous = np.array([pt[0], pt[1], 1])
                    pt_transformed = cumulative_transform @ pt_homogeneous  # Multiplication matrice
                    transformed_points.append([pt_transformed[0], pt_transformed[1]])
                # Simplification des points (Ramer-Douglas-Peucker)
                simplified_points = rdp(transformed_points, epsilon=0.01)
                # Ajout des points transformés à la liste globale
                all_points.extend(simplified_points)
                shapes_data.append(simplified_points)
                shape_keys.append((path_idx, sub_idx))
                # On stocke dans shape_history toutes les infos utiles pour le debug et la génération du SVG de résumé
                shape_history[(path_idx, sub_idx)] = {
                    'svg_path_idx': path_idx,
                    'svg_sub_idx': sub_idx,
                    'svg_attr': dict(svgpathtools_attr),
                    'sampled_points': transformed_points,
                    'simplified_points': simplified_points,
                    'cq_wire_index': None,
                }
                logger.debug(f"Path {path_idx}, subpath {sub_idx}, nb_points: {len(simplified_points)}, type: path, attr: {svgpathtools_attr}")

    logger.debug(f"shape_history keys: {list(shape_history.keys())}")
    for k, v in shape_history.items():
        if isinstance(k, tuple):
            logger.debug(f"shape_history[{k}]: nb_points={len(v.get('simplified_points', []))}, type={k[0]}")

    if not all_points:
        logger.error("Aucun chemin SVG trouvé avec fill ou stroke pour créer le moule.")
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
    wire_to_shape = {}
    for shape_idx, shape_points in enumerate(shapes_data):
        retry = True
        while retry:
            scaled_points = [((x - min_x) * scale, (y - min_y) * scale) for x, y in shape_points]
            filtered_points = filter_points(scaled_points)
            # On prépare la liste des points effectivement utilisés pour le wire (après fermeture)
            # S'assurer que chaque point est un tuple plat (x, y)
            extrusion_points = [tuple(map(float, pt)) for pt in filtered_points]
            if len(extrusion_points) > 2:
                first_pt = np.array(extrusion_points[0])
                last_pt = np.array(extrusion_points[-1])
                dist = np.linalg.norm(first_pt - last_pt)
                if dist > 1e-6:
                    extrusion_points.append(tuple(map(float, extrusion_points[0])))
            if len(extrusion_points) > 2:
                try:
                    temp_wp = cq.Workplane("XY").polyline(extrusion_points).close()
                    if len(temp_wp.wires().objects) == 1:
                        single_wire = temp_wp.wires().objects[0]
                        verts = list(single_wire.Vertices())
                        if single_wire.isValid() and np.linalg.norm(np.array([verts[0].X, verts[0].Y]) - np.array([verts[-1].X, verts[-1].Y])) < 1e-6:
                            cq_wires.append(single_wire)
                            shape_history[shape_keys[shape_idx]]['cq_wire_index'] = len(cq_wires) - 1
                            wire_to_shape[len(cq_wires) - 1] = shape_keys[shape_idx]
                            # Ajout des points utilisés pour l'extrusion
                            shape_history[shape_keys[shape_idx]]['extrusion_points'] = list(extrusion_points)
                            retry = False
                            continue
                    try:
                        poly_wire = cq.Wire.makePolygon([cq.Vector(x, y) for x, y in extrusion_points])
                        if poly_wire.isValid():
                            cq_wires.append(poly_wire)
                            shape_history[shape_keys[shape_idx]]['cq_wire_index'] = len(cq_wires) - 1
                            wire_to_shape[len(cq_wires) - 1] = shape_keys[shape_idx]
                            shape_history[shape_keys[shape_idx]]['extrusion_points'] = list(extrusion_points)
                            retry = False
                            continue
                    except Exception:
                        pass
                except Exception:
                    pass
            if retry:
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
            logger.info(f"Offset de {margin}mm appliqué à l'enveloppe convexe pour la marge.")
        else:
            logger.warning(f"Offset de {margin}mm échoué, utilisation de l'enveloppe convexe brute.")
    except Exception as e:
        logger.warning(f"Erreur lors de l'offset de l'enveloppe convexe : {e}. Utilisation de l'enveloppe brute.")

    # Diagnostic wire
    logger.debug(f"base_outline_wire type: {type(base_outline_wire)}, isValid: {base_outline_wire.isValid() if hasattr(base_outline_wire, 'isValid') else 'N/A'}, isClosed: {base_outline_wire.IsClosed() if hasattr(base_outline_wire, 'IsClosed') else 'N/A'}")
    if not base_outline_wire.isValid() or not base_outline_wire.IsClosed():
        logger.debug("Wire offset non valide ou non fermé, tentative de reconstruction via makePolygon.")
        pts = [v.toTuple() for v in base_outline_wire.Vertices()]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        base_outline_wire = cq.Wire.makePolygon([cq.Vector(x, y) for x, y in pts])
        logger.debug(f"Wire reconstruit: isValid: {base_outline_wire.isValid()}, isClosed: {base_outline_wire.IsClosed()}")

    # Création de la base
    base_wp = cq.Workplane("XY").add(base_outline_wire).toPending().extrude(base_thickness)
    if not (base_wp.val().isValid() and base_wp.val().ShapeType() == "Solid"):
        logger.error("La base du moule (contour SVG) n'est pas un solide valide.")
        raise ValueError("La base du moule (contour SVG) n'est pas un solide valide.")
    base_shape = base_wp.val()

    # --- Génération de la bordure en anneau ---
    try:
        border_outer_wires = base_outline_wire.offset2D(border_thickness, "arc")
        if isinstance(border_outer_wires, list) and len(border_outer_wires) > 0:
            border_outer_wire = border_outer_wires[0]
            logger.info(f"Offset extérieur de {border_thickness}mm appliqué pour la bordure.")
        else:
            logger.warning(f"Offset extérieur échoué, pas de bordure.")
            border_outer_wire = None
    except Exception as e:
        logger.warning(f"Erreur lors de l'offset extérieur pour la bordure : {e}. Pas de bordure.")
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
        logger.info("Bordure en anneau générée et fusionnée à la base.")
    else:
        base_mold = base_shape
        logger.info("Pas de bordure générée, seule la base est utilisée.")

    # --- Export STL de la base avant gravure dans debug_stl/ ---
    base_stl_path = os.path.join(debug_dir, os.path.basename(base_stl_name))
    if export_base_stl:
        cq.exporters.export(base_mold, base_stl_path)
        logger.info(f"STL de la base (avec bordure) exporté : {base_stl_path}")
    return base_mold

def try_apply_engraving(engraving_obj, method_label, group_idx, mold, base_thickness, export_steps, debug_dir, engraved_indices):
    """
    Tente d'appliquer un engraving (loft ou extrusion), effectue le cut, exporte le STL si demandé, et met à jour les indices gravés.
    Retourne (success, new_mold)
    """
    engravings = flatten_cq_solids(engraving_obj)
    logger.debug(f"Types des objets retournés par flatten_cq_solids ({method_label}): {[type(e) for e in engravings]}")
    for engraving_solid in engravings:
        if hasattr(engraving_solid, 'isValid') and engraving_solid.isValid() and hasattr(engraving_solid, 'ShapeType') and engraving_solid.ShapeType() in ["Solid", "Compound"]:
            engraving_solid = engraving_solid.translate((0, 0, base_thickness))
            try:
                new_mold = mold.cut(engraving_solid)
                if export_steps:
                    step_stl_path = os.path.join(debug_dir, f"step_{group_idx}.stl")
                    cq.exporters.export(new_mold, step_stl_path)
                    logger.info(f"Export STL intermédiaire réussi ({method_label}) : {step_stl_path}")
                engraved_indices.append(group_idx)
                logger.info(f"Groupe {group_idx} gravé avec succès ({method_label}).")
                return True, new_mold
            except Exception as ce:
                logger.warning(f"Erreur lors du cut avec l'extrusion du groupe {group_idx} ({method_label}) : {ce}")
    return False, mold

def engrave_polygons(mold, svg_wires, shape_history, base_thickness, engrave_depth, export_steps, debug_dir,
                      original_svg_path=None):
    """
    Grave les polygones sur le moule, met à jour shape_history['engraved'] pour chaque shape.
    Retourne le moule gravé et la liste des indices gravés.
    Génère un SVG de résumé à chaque étape de gravure.
    """

    engraved_indices = []
    grouped_wires = group_wires_by_inclusion(svg_wires, shape_history)
    wire_to_shape = shape_history.get('wire_to_shape', {})

    group_counter = 0
    for group_idx, wire_group in grouped_wires:
        try:
            logger.info(f"--- Gravure polygone {group_idx} (groupe de {len(wire_group)} wires) ---")
            for wire_index, wire in enumerate(wire_group):
                try:
                    global_wire_index = svg_wires.index(wire)
                except ValueError:
                    global_wire_index = None
                shape_info = wire_to_shape.get(global_wire_index, None)
                if shape_info is not None:
                    hist = shape_history[shape_info]
                    logger.debug(f"  Wire {wire_index} (global idx {global_wire_index}): SVG path_idx={hist['svg_path_idx']}, sub_idx={hist['svg_sub_idx']}, nb_points={len(hist['simplified_points'])}")
                else:
                    logger.debug(f"  Wire {wire_index} (global idx {global_wire_index}): [shape SVG non trouvée]")
                logger.debug(f"    isClosed: {wire.IsClosed()}, isValid: {wire.isValid()}, nb sommets: {len(list(wire.Vertices()))}")

            # 1. Tenter d'abord le loft avec dépouille
            draft_angle_deg = 15  # Angle de dépouille en degrés (modifiable)
            engraving_loft = loft_with_draft(wire_group, draft_angle_deg, engrave_depth)
            logger.debug(f"Type engraving retourné par loft_with_draft: {type(engraving_loft)}")
            success, mold = try_apply_engraving(
                engraving_loft, "loft", group_counter,
                mold, base_thickness, export_steps,
                debug_dir, engraved_indices
            )
            if not success:
                logger.warning(f"Loft du groupe {group_counter} n'a pas produit de solide valide.")
                # 2. Si le loft échoue, tenter l'extrusion directe
                logger.info(f"--- Tentative d'extrusion directe pour le groupe {group_counter} ---")
                try:
                    face_result = cq.Face.makeFromWires(wire_group[0], wire_group[1:])
                except Exception as e:
                    logger.warning(f"Erreur makeFromWires sur groupe {group_counter}: {e}")
                    mark_wires_engraved(wire_group, False, svg_wires, shape_history, group_counter)
                    group_counter += 1
                    continue
                if isinstance(face_result, list):
                    if not face_result:
                        logger.warning(f"Polygone {group_counter} : makeFromWires a retourné une liste vide.")
                        group_counter += 1
                        continue
                    face = face_result[0]
                else:
                    face = face_result
                engraving_extrude = cq.Workplane("XY").add(face).extrude(-engrave_depth)
                logger.debug(f"Type engraving après extrusion directe: {type(engraving_extrude)}")
                success, mold = try_apply_engraving(
                    engraving_extrude, "extrusion", group_counter,
                    mold, base_thickness, export_steps,
                    debug_dir, engraved_indices
                )
                if not success:
                    logger.warning(f"Extrusion du groupe {group_counter} n'a pas produit de solide valide.")
                    mark_wires_engraved(wire_group, False, svg_wires, shape_history, group_counter)
                    raise ValueError(f"Le groupe {group_counter} n'a pas pu être gravé avec succès.")
                else:
                    mark_wires_engraved(wire_group, True, svg_wires, shape_history, group_counter)
            else:
                mark_wires_engraved(wire_group, True, svg_wires, shape_history, group_counter)
        except Exception as e:
            logger.error(f"Erreur lors de la gravure du groupe {group_counter}: {e}")

        # Génération du SVG de résumé après chaque étape de gravure
        if original_svg_path is not None:
            current_shape_keys = []
            for w in wire_group:
                global_wire_index = get_global_wire_index(w, svg_wires)
                if global_wire_index is not None and global_wire_index in wire_to_shape:
                    current_shape_keys.append(wire_to_shape[global_wire_index])
            if current_shape_keys:
                summary_svg_path = os.path.join(debug_dir, f"step_{group_idx}_summary.svg")
                generate_summary_svg(original_svg_path, current_shape_keys, summary_svg_path, shape_history=shape_history)
            else:
                logger.warning(f"Aucune shape_key trouvée pour le groupe {group_counter}, le SVG de résumé n'a pas été généré.")
        group_counter += 1
    return mold, engraved_indices

def generate_cadquery_mold(svg_file, max_dim, base_thickness=BASE_THICKNESS, border_height=BORDER_HEIGHT,
                           border_thickness=BORDER_THICKNESS, engrave_depth=ENGRAVE_DEPTH, margin=MARGE,
                           export_base_stl=True, base_stl_name="moule_base.stl", export_steps=False,
                           keep_debug_files=False):
    # Détermination du nom du dossier de debug à partir du nom du fichier SVG
    svg_basename = os.path.splitext(os.path.basename(svg_file))[0]
    debug_dir = f"debug_{svg_basename}"
    os.makedirs(debug_dir, exist_ok=True)

    logger.info(f"Normalisation du SVG : {svg_file}")
    normalized_svg_file = normalize_svg_fill(svg_file, debug_dir=debug_dir)

    # Récupération des wires et de l'historique des shapes
    svg_wires, shape_history = svg_to_cadquery_wires(normalized_svg_file, max_dim)

    # Calcul du rapport d'agrandissement/réduction et du min_x/min_y pour conserver l'échelle d'origine
    # On récupère tous les points utilisés pour le SVG
    all_points = []
    for k in shape_history:
        if isinstance(k, tuple):
            pts = shape_history[k].get('simplified_points')
            if pts:
                all_points.extend(pts)
    if all_points:
        xs = [pt[0] for pt in all_points]
        ys = [pt[1] for pt in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        svg_width = max_x - min_x
        svg_height = max_y - min_y
        if svg_width == 0 and svg_height == 0:
            scale = 1.0
        elif svg_width == 0:
            scale = (max_dim - 2 * MARGE) / svg_height
        elif svg_height == 0:
            scale = (max_dim - 2 * MARGE) / svg_width
        else:
            scale = (max_dim - 2 * MARGE) / max(svg_width, svg_height)
        # On sauvegarde dans shape_history
        shape_history['svg_scale'] = scale
        shape_history['svg_min_x'] = min_x
        shape_history['svg_min_y'] = min_y
        shape_history['svg_width'] = svg_width
        shape_history['svg_height'] = svg_height
    else:
        shape_history['svg_scale'] = None
        shape_history['svg_min_x'] = None
        shape_history['svg_min_y'] = None
        shape_history['svg_width'] = None
        shape_history['svg_height'] = None

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
        original_svg_path=svg_file
    )

    # Génération d'un SVG de résumé global de toutes les formes (gravées ou non)
    all_shape_keys = [k for k in shape_history if isinstance(k, tuple)]
    if all_shape_keys:
        global_summary_svg = os.path.join(debug_dir, f"summary_{svg_basename}_final.svg")
        generate_summary_svg(svg_file, all_shape_keys, global_summary_svg, shape_history=shape_history)
        logger.info(f"Résumé global SVG généré : {global_summary_svg}")
    else:
        logger.warning("Aucune forme trouvée, pas de SVG de résumé global généré.")

    # Suppression du dossier de debug si demandé
    if not keep_debug_files:
        shutil.rmtree(debug_dir, ignore_errors=True)

    return mold, engraved_indices, shape_history

def generate_summary_svg(original_svg_path, shape_keys, output_svg_name, shape_history=None):
    """
    Génère un SVG de résumé en dessinant explicitement chaque shape (polygone) à partir des points simplifiés de shape_history,
    coloré en noir si engraved, rouge sinon. Le SVG généré est vierge et ne contient que les polygones de l'étape.
    """
    logger.info(f"Génération du résumé SVG pour {output_svg_name}")
    logger.debug(f"shape_keys à dessiner: {shape_keys}")
    if shape_history is not None:
        for k in shape_keys:
            hist = shape_history.get(k, None)
            if hist is not None:
                logger.debug(f"  - shape_key: {k}, nb_points: {len(hist.get('simplified_points', []))}, attr: {hist.get('svg_attr', {})}")
    # Lit l'ancien SVG pour récupérer ses attributs (width, height, viewBox)
    original_tree = ET.parse(original_svg_path)
    original_root = original_tree.getroot()
    
    # Crée un nouvel élément SVG racine, en copiant explicitement viewBox, width, height, xmlns
    ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', ns) # Namespace par défaut
    new_root_attribs = {}
    # Copie viewBox, width, height si présents
    for attr in ['viewBox', 'width', 'height']:
        if attr in original_root.attrib:
            new_root_attribs[attr] = original_root.attrib[attr]
    # Ajoute le namespace
    new_root_attribs['xmlns'] = ns
    new_root = ET.Element('svg', new_root_attribs)

    # Création d'un groupe pour les polygones sélectionnés
    group = ET.Element('g', {'id': 'summary_polygons'})
    new_root.append(group)


    # Nouvelle logique : dessiner tous les paths, même non gravés, en rouge (remplissage) si la gravure a échoué.
    if shape_history is not None:
        from collections import defaultdict
        grouped = defaultdict(list)
        for (path_idx, sub_idx) in shape_keys:
            grouped[path_idx].append((path_idx, sub_idx))

        # Récupération des infos d'échelle pour remettre les polygones dans la viewBox d'origine
        svg_scale = shape_history.get('svg_scale', None)
        svg_min_x = shape_history.get('svg_min_x', 0)
        svg_min_y = shape_history.get('svg_min_y', 0)
        # Si pas d'échelle, on ne fait pas de correction
        def to_original_coords(pt):
            if svg_scale and svg_scale != 0:
                x = pt[0] / svg_scale + svg_min_x
                y = pt[1] / svg_scale + svg_min_y
                return (x, y)
            else:
                return pt

        for path_idx, subkeys in grouped.items():
            paths_d = ''
            engraved = True  # Par défaut
            for (pidx, sidx) in subkeys:
                hist = shape_history.get((pidx, sidx), None)
                if hist is not None:
                    # On tente d'utiliser extrusion_points, sinon simplified_points, sinon sampled_points
                    pts = hist.get('extrusion_points')
                    if not pts or len(pts) < 3:
                        pts = hist.get('simplified_points')
                    if not pts or len(pts) < 3:
                        pts = hist.get('sampled_points')
                    if pts and len(pts) >= 3:
                        # On prend le statut engraved du contour extérieur (premier subkey)
                        if (pidx, sidx) == subkeys[0]:
                            engraved = hist.get('engraved', False)
                        # Conversion inverse de l'échelle pour chaque point
                        pts_orig = [to_original_coords(pt) for pt in pts]
                        paths_d += 'M ' + ' '.join(f'{x},{y}' for x, y in pts_orig) + ' Z '
            if paths_d:
                if engraved:
                    style = 'fill:black;stroke:none'
                else:
                    style = 'fill:red;stroke:none'
                path_elem = ET.Element('path', {
                    'd': paths_d.strip(),
                    'style': style,
                    'fill-rule': 'evenodd',
                    'data-pathidx': str(path_idx),
                })
                group.append(path_elem)
            else:
                # Si aucun point utilisable, on génère un path vide en marron (stroke)
                path_elem = ET.Element('path', {
                    'd': '',
                    'style': 'fill:brown;stroke:none',
                    'data-pathidx': str(path_idx),
                })
                group.append(path_elem)
    else:
        logger.warning("Aucune shape_history fournie pour générer le SVG de résumé.")

    # Écriture du nouveau fichier SVG (toujours à la fin, une seule fois)
    new_tree = ET.ElementTree(new_root)
    new_tree.write(output_svg_name, encoding='utf-8', xml_declaration=True)
    logger.info(f"SVG généré : {output_svg_name}")


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
        logger.debug(f"[flatten_cq_solids] Workplane.val() type: {type(val)}")
        solids.extend(flatten_cq_solids(val))
    elif hasattr(obj, 'vals') and callable(obj.vals):
        # Workplane avec plusieurs objets
        for v in obj.vals():
            logger.debug(f"[flatten_cq_solids] Workplane.vals() type: {type(v)}")
            solids.extend(flatten_cq_solids(v))
    else:
        logger.debug(f"[flatten_cq_solids] Objet ignoré (ni CQ, ni Workplane, ni liste): {type(obj)} -> {obj}")
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
                # Affichage des indices globaux des wires du groupe
                global_indices = [get_global_wire_index(wires_polygons[outer_index][0], wires)] + [get_global_wire_index(wires_polygons[j][0], wires) for j in inners_indices]
                logger.debug(f"Group formed. Outer wire index: {outer_index} (global: {global_indices[0]}), inner wires idxs: {[j for j in inners_indices]} (globals: {global_indices[1:]})")
            else:
                # Aucun wire restant ne contient d'autres, on ajoute les restants seuls
                for idx in range(len(wires_polygons)):
                    if idx not in used_indices:
                        all_groups.append((path_idx, [wires_polygons[idx][0]]))
                        used_indices.add(idx)
                        global_idx = get_global_wire_index(wires_polygons[idx][0], wires)
                        logger.debug(f"Group formed. Single wire: {wires_polygons[idx][0]} (no inners), global index: {global_idx}")
    # Vérification que tous les path_idx ont été utilisés
    used_path_indices = {g[0] for g in all_groups}
    for path_idx in path_idx_to_wire_indices.keys():
        if path_idx not in used_path_indices:
            logger.warning(f"Path index {path_idx} not used in any group, possible issue with SVG structure.")
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
        logger.warning("Polygone offset non simple, possible auto-intersection")
    return offset_points

def get_scaled_wire_from_wire(wire, offset, centroid_direction, auto_simplify=True):
    """
    Génère un wire décalé (scaled) à partir d'un wire d'origine, selon la normale locale et un offset.
    Si auto_simplify=True (par défaut), réduit automatiquement l'offset tant que le polygone n'est pas simple (pas d'auto-intersection).
    Args:
        wire (cq.Wire): Wire d'origine
        offset (float): Distance de décalage
        centroid_direction (int): +1 pour outer, -1 pour inner
        auto_simplify (bool): Si True, réduit l'offset automatiquement si le polygone n'est pas simple
    Returns:
        cq.Wire: Wire décalé
    """
    wire_points = [(vertex.X, vertex.Y) for vertex in wire.Vertices()]
    n_points = len(wire_points)
    # Génération initiale des points offsetés
    offset_points = offset_polygon_along_normals(wire_points, offset, centroid_direction=centroid_direction)
    shapely_poly = ShapelyPolygon(offset_points)

    # Si le polygone est simple ou si c'est l'outer, on retourne directement
    if centroid_direction == 1 and shapely_poly.is_simple:
        return cq.Wire.makePolygon([cq.Vector(x, y) for x, y in offset_points])

    # Si auto_simplify est activé, on réduit l'offset automatiquement tant que le polygone n'est pas simple
    if auto_simplify:
        offset_attempt = float(offset)
        min_offset = 1e-5  # Valeur minimale d'offset pour éviter de tourner en boucle
        max_auto_attempts = 16  # Nombre maximum de tentatives pour éviter une boucle infinie
        attempt_counter = 0
        while not shapely_poly.is_simple and abs(offset_attempt) > min_offset and attempt_counter < max_auto_attempts:
            attempt_counter += 1
            offset_attempt *= 0.5  # Réduction progressive de l'offset
            offset_points = offset_polygon_along_normals(wire_points, offset_attempt, centroid_direction=centroid_direction)
            shapely_poly = ShapelyPolygon(offset_points)
        # Si on a trouvé un polygone simple, on le retourne
        if shapely_poly.is_simple:
            # On resample si besoin pour garder le même nombre de points
            if len(offset_points) < len(wire_points):
                resampled = resample_polygon(list(shapely_poly.exterior.coords), n_points)
                aligned = align_resampled_to_reference(resampled, wire_points)
            else:
                aligned = offset_points
            return cq.Wire.makePolygon([cq.Vector(x, y) for x, y in aligned])
        # Si aucune solution trouvée, offset zéro
        offset_points_zero = offset_polygon_along_normals(wire_points, 0, centroid_direction=centroid_direction)
        shapely_zero = ShapelyPolygon(offset_points_zero)
        resampled = resample_polygon(list(shapely_zero.exterior.coords), n_points)
        aligned = align_resampled_to_reference(resampled, wire_points)
        return cq.Wire.makePolygon([cq.Vector(x, y) for x, y in aligned])

    # Sinon, on propose à l'utilisateur de gérer l'auto-intersection (logique interactive)
    max_attempts = 8
    for attempt in range(max_attempts):
        if shapely_poly.is_simple:
            if len(offset_points) < len(wire_points):
                resampled = resample_polygon(list(shapely_poly.exterior.coords), n_points)
                aligned = align_resampled_to_reference(resampled, wire_points)
            else:
                aligned = offset_points
                return cq.Wire.makePolygon([cq.Vector(x, y) for x, y in aligned])
        # Affichage matplotlib avant/après simplification avec boutons interactifs
        try:
            user_choice = {'value': None}

            def on_button_clicked(event, choice):
                user_choice['value'] = choice
                plt.close()

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].set_title('Avant simplification')
            x, y = shapely_poly.exterior.xy
            axs[0].plot(x, y, 'r-', label='Original')
            axs[0].fill(x, y, alpha=0.2, color='red')
            axs[0].axis('equal')
            # Simplification avec buffer(0)
            simple_poly = shapely_poly.buffer(0)
            axs[1].set_title('Après simplification (buffer(0))')
            if isinstance(simple_poly, MultiPolygon):
                for poly in simple_poly.geoms:
                    x2, y2 = poly.exterior.xy
                    axs[1].plot(x2, y2, 'g-')
                    axs[1].fill(x2, y2, alpha=0.2, color='green')
            elif isinstance(simple_poly, ShapelyPolygon):
                x2, y2 = simple_poly.exterior.xy
                axs[1].plot(x2, y2, 'g-')
                axs[1].fill(x2, y2, alpha=0.2, color='green')
            axs[1].axis('equal')
            plt.suptitle(f"Wire offset inner - tentative {attempt+1}")

            # Ajout des boutons sous la figure
            button_labels = [
                "Diminueer l'offset",
                "Simplification OK",
                "Offset zéro",
                "Ignorer ce wire"
            ]
            actions = ['retry', 'simplify', 'zero', 'ignore']
            button_axes = []
            buttons = []
            n_buttons = len(button_labels)
            for i, label in enumerate(button_labels):
                ax_btn = plt.axes([0.1 + 0.2*i, 0.01, 0.18, 0.06])
                btn = Button(ax_btn, label)
                btn.on_clicked(lambda event, c=actions[i]: on_button_clicked(event, c))
                button_axes.append(ax_btn)
                buttons.append(btn)

            plt.show()

            # Après fermeture de la fenêtre, lire le choix
            choice = user_choice['value']
            if choice == 'simplify':
                # On prend le polygone simplifié (le plus grand si MultiPolygon)
                if isinstance(simple_poly, MultiPolygon):
                    largest = max(simple_poly.geoms, key=lambda p: p.area)
                    simple_poly = largest
                simple_points = list(simple_poly.exterior.coords)
                resampled = resample_polygon(simple_points, n_points)
                aligned = align_resampled_to_reference(resampled, wire_points)
                return cq.Wire.makePolygon([cq.Vector(x, y) for x, y in aligned])
            elif choice == 'zero':
                print("Offset mis à 0 pour ce wire sur demande utilisateur.")
                offset_points_zero = offset_polygon_along_normals(wire_points, 0, centroid_direction=centroid_direction)
                shapely_zero = ShapelyPolygon(offset_points_zero)
                resampled = resample_polygon(list(shapely_zero.exterior.coords), n_points)
                aligned = align_resampled_to_reference(resampled, wire_points)
                return cq.Wire.makePolygon([cq.Vector(x, y) for x, y in aligned])
            elif choice == 'ignore':
                print("Wire ignoré sur demande utilisateur.")
                return None
            else:
                # Retenter avec un offset plus faible
                offset = offset * 0.5
                offset_points = offset_polygon_along_normals(wire_points, offset, centroid_direction=centroid_direction)
                shapely_poly = ShapelyPolygon(offset_points)
        except Exception as e:
            print(f"Erreur lors de l'affichage ou de la simplification matplotlib : {e}")
            # Si erreur, on retente avec offset plus faible
            offset = offset * 0.5
            offset_points = offset_polygon_along_normals(wire_points, offset, centroid_direction=centroid_direction)
            shapely_poly = ShapelyPolygon(offset_points)
    # Si aucune solution trouvée, offset zéro
    print("Offset mis à 0 pour ce wire après toutes les tentatives.")
    offset_points_zero = offset_polygon_along_normals(wire_points, 0, centroid_direction=centroid_direction)
    shapely_zero = ShapelyPolygon(offset_points_zero)
    resampled = resample_polygon(list(shapely_zero.exterior.coords), n_points)
    aligned = align_resampled_to_reference(resampled, wire_points)
    return cq.Wire.makePolygon([cq.Vector(x, y) for x, y in aligned])


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
        offset_initial = abs(depth) * math.tan(math.radians(draft_angle_deg))
        logger.debug(f"[loft_with_draft] Début du loft individuel. Nombre de wires: {len(wire_group)}")
        lofted_solids = []
        # Parcours de chaque wire du groupe (contour principal + trous)
        for wire_index, wire in enumerate(wire_group):
            # Sens de l'offset : +1 pour outer, -1 pour inners
            centroid_direction = 1 if wire_index == 0 else -1
            # Outer : une seule tentative avec l'offset initial, inners : gestion interactive dans get_scaled_wire_from_wire
            scaled_wire = get_scaled_wire_from_wire(wire, offset_initial, centroid_direction)
            # Si l'utilisateur a choisi d'ignorer ce wire, on saute la création
            if scaled_wire is None:
                logger.info(f"Wire {wire_index} ignoré, aucune gravure ne sera réalisée pour ce trou.")
                continue
            # Création des faces (top = wire décalé, bottom = wire original)
            try:
                face_top = cq.Face.makeFromWires(scaled_wire)
                face_bottom = cq.Face.makeFromWires(wire)
            except Exception as e:
                logger.warning(f"[loft_with_draft] Echec de makeFromWires (individuel) : {e}")
                continue  # Passe au wire suivant si échec
            # Placement des faces à la bonne hauteur Z
            face_top = face_top.translate((0, 0, 0))
            face_bottom = face_bottom.translate((0, 0, -depth))
            # Création du loft entre les deux faces
            workplane = cq.Workplane("XY").add([face_top, face_bottom])
            try:
                lofted_solid = workplane.loft(combine=True)
                logger.debug(f"[loft_with_draft] Loft individuel réussi pour wire {wire_index}.")
                lofted_solids.append(lofted_solid)
            except Exception as e:
                logger.warning(f"[loft_with_draft] Echec du loft individuel pour wire {wire_index}: {e}")
                return None  # Passe au wire suivant si échec
        # Combine les solides : outer (index 0) en add, inners (>0) en cut
        if not lofted_solids:
            logger.warning("[loft_with_draft] Aucun solide généré.")
            return None
        # Le premier solide (contour principal) sert de base
        result_solid = lofted_solids[0]
        # Les suivants (trous) sont soustraits
        for inner_solid in lofted_solids[1:]:
            try:
                result_solid = result_solid.cut(inner_solid)
            except Exception as e:
                logger.warning(f"[loft_with_draft] Echec du cut pour inner : {e}")
        return result_solid
    except Exception as e:
        logger.warning(f"[loft_with_draft] Echec du loft avec dépouille (individuel) : {e}")
        return None

def get_global_wire_index(wire, svg_wires):
    """
    Retourne l'index global du wire dans svg_wires, ou None si non trouvé.
    """
    try:
        return svg_wires.index(wire)
    except ValueError:
        return None

def mark_wires_engraved(wire_group, engraved, svg_wires, shape_history, group_idx):
    """
    Marque les wires du groupe comme gravés ou non gravés dans shape_history.
    """
    for w in wire_group:
        global_wire_index = get_global_wire_index(w, svg_wires)
        if global_wire_index is not None:
            for shape_key, hist in shape_history.items():
                if isinstance(shape_key, tuple) and hist.get('cq_wire_index') == global_wire_index:
                    hist['engraved'] = engraved
        else:
            logger.warning(f"Impossible de trouver le wire: {w} dans svg_wires pour le groupe {group_idx}.")
