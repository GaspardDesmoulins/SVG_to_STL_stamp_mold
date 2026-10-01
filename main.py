import argparse
from pathlib import Path

from moule_svg_cadquery import MoldGenerationError, generate_cadquery_mold
import cadquery as cq
from settings import BASE_THICKNESS, BORDER_HEIGHT, BORDER_THICKNESS, \
    MARGE, ENGRAVE_DEPTH, MAX_DIMENSION

def main():
    parser = argparse.ArgumentParser(description="Génère un moule à silicone depuis un SVG avec CadQuery.")
    parser.add_argument("--svg", required=True, help="Chemin du fichier SVG")
    parser.add_argument("--size", type=float, default=MAX_DIMENSION, help="Taille max du moule (mm)")
    parser.add_argument("--output", default="moule_cadquery.stl", help="Fichier de sortie STL")
    engraving_mode = parser.add_mutually_exclusive_group()
    engraving_mode.add_argument(
        "--stepped",
        action="store_const",
        const="stepped",
        dest="engraving_mode",
        help="Utilise la gravure voxelisée par couches (mode par défaut).",
    )
    engraving_mode.add_argument(
        "--classic",
        action="store_const",
        const="classic",
        dest="engraving_mode",
        help="Utilise le loft avec dépouille, moins robuste pour les contours complexes.",
    )
    parser.set_defaults(engraving_mode="stepped")
    parser.add_argument("--layer-thickness", type=float, default=0.1, help="Épaisseur d'une couche voxelisée (mm)")
    parser.add_argument("--pixel-size", type=float, default=0.1, help="Résolution de la grille voxelisée (mm)")
    parser.add_argument("--growth-per-layer", type=int, default=1, help="Croissance des contours par couche (pixels)")
    parser.add_argument("--export-steps", action="store_true", help="Exporte les STL intermédiaires dans le dossier de debug")
    parser.add_argument("--keep-debug-files", action="store_true", help="Conserver le répertoire de debug et les fichiers intermédiaires")
    args = parser.parse_args()

    svg_path = Path(args.svg)
    output_path = Path(args.output)
    if not svg_path.is_file():
        parser.error(f"le fichier SVG n'existe pas : {svg_path}")
    if args.size <= 0:
        parser.error("--size doit être strictement positif")
    if args.layer_thickness <= 0 or args.pixel_size <= 0:
        parser.error("--layer-thickness et --pixel-size doivent être strictement positifs")
    if args.growth_per_layer < 0:
        parser.error("--growth-per-layer ne peut pas être négatif")
    if output_path.parent != Path(".") and not output_path.parent.exists():
        parser.error(f"le répertoire de sortie n'existe pas : {output_path.parent}")

    try:
        mold, engraved_indices, shape_history = generate_cadquery_mold(
            str(svg_path),
            args.size,
            base_thickness=BASE_THICKNESS,
            border_height=BORDER_HEIGHT,
            border_thickness=BORDER_THICKNESS,
            engrave_depth=ENGRAVE_DEPTH,
            margin=MARGE,
            export_base_stl=True,
            export_steps=args.export_steps,
            engraving_mode=args.engraving_mode,
            base_stl_name="moule_base.stl",
            keep_debug_files=args.keep_debug_files,
            layer_thickness_mm=args.layer_thickness,
            pixel_size_mm=args.pixel_size,
            growth_per_layer_px=args.growth_per_layer,
        )
        cq.exporters.export(mold, str(output_path))
        print(f"Moule CadQuery généré : {output_path}")
    except MoldGenerationError as error:
        parser.exit(1, f"Échec de génération du moule : {error}\n")
    except Exception as error:
        parser.exit(1, f"Erreur inattendue lors de la génération du moule : {error}\n")


if __name__ == "__main__":
    main()
