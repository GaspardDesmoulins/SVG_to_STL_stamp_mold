import os
import cadquery as cq
import numpy as np
from scipy.spatial import ConvexHull
from settings import BASE_THICKNESS, BORDER_HEIGHT, BORDER_THICKNESS, MARGE, MAX_DIMENSION

def svg_to_cadquery_wires(svg_file, max_dimension=MAX_DIMENSION):
    # Cette fonction doit être adaptée ou importée depuis le code existant
    raise NotImplementedError("À implémenter : conversion SVG -> wires CadQuery.")


def create_mold_base_from_svg(svg_file, max_dim=MAX_DIMENSION, base_thickness=BASE_THICKNESS, border_height=BORDER_HEIGHT,
                             border_thickness=BORDER_THICKNESS, margin=MARGE, export_base_stl=True, base_stl_name="moule_base.stl"):
    """
    Génère la base du moule (solide extrudé) à partir d'un SVG, sans gravure, pour une extrusion couche par couche.
    Retourne le solide de base (mold) et le wire de contour utilisé.
    """
    # Extraction des wires à partir du SVG
    svg_wires, _ = svg_to_cadquery_wires(svg_file, max_dim)

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

    # Création de la base avec marge
    base_outline_wire = cq.Wire.makePolygon([cq.Vector(x, y) for x, y in outline + [outline[0]]])
    # Offset pour la marge
    try:
        base_offset_wires = base_outline_wire.offset2D(margin, "arc")
        if isinstance(base_offset_wires, list) and len(base_offset_wires) > 0:
            base_outline_wire = base_offset_wires[0]
    except Exception:
        pass

    # Création de la base
    base_wp = cq.Workplane("XY").add(base_outline_wire).toPending().extrude(base_thickness)
    base_shape = base_wp.val()

    # Génération de la bordure en anneau
    try:
        border_outer_wires = base_outline_wire.offset2D(border_thickness, "arc")
        if isinstance(border_outer_wires, list) and len(border_outer_wires) > 0:
            border_outer_wire = border_outer_wires[0]
        else:
            border_outer_wire = None
    except Exception:
        border_outer_wire = None

    border_inner_wire = base_outline_wire

    if border_outer_wire is not None:
        border_outer_wp = cq.Workplane("XY").add(border_outer_wire).toPending().extrude(base_thickness + border_height)
        border_outer_shape = border_outer_wp.val()
        border_inner_wp = cq.Workplane("XY").add(border_inner_wire).toPending().extrude(base_thickness + border_height)
        border_inner_shape = border_inner_wp.val()
        border_ring = border_outer_shape.cut(border_inner_shape)
        base_mold = base_shape.fuse(border_ring)
    else:
        base_mold = base_shape

    # Export STL de la base
    if export_base_stl:
        cq.exporters.export(base_mold, base_stl_name)
    return base_mold, base_outline_wire
