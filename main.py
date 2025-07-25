import argparse
from moule_svg_cadquery import generate_cadquery_mold, generate_summary_svg
import os
import cadquery as cq
from settings import BASE_THICKNESS, BORDER_HEIGHT, BORDER_THICKNESS, \
    MARGE, ENGRAVE_DEPTH, MAX_DIMENSION

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère un moule à silicone depuis un SVG avec CadQuery.")
    parser.add_argument("--svg", help="Chemin du fichier SVG")
    parser.add_argument("--size", type=float, default=MAX_DIMENSION, help="Taille max du moule (mm)")
    parser.add_argument("--output", default="moule_cadquery.stl", help="Fichier de sortie STL")
    parser.add_argument("--no-interactive", action="store_true", help="Désactive le mode interactif")
    parser.add_argument("--keep-debug-files", action="store_true", help="Conserver le répertoire de debug et les fichiers intermédiaires")
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
            base_thickness=BASE_THICKNESS,
            border_height=BORDER_HEIGHT,
            border_thickness=BORDER_THICKNESS,
            engrave_depth=ENGRAVE_DEPTH,
            margin=MARGE,
            export_base_stl=True,
            base_stl_name="moule_base.stl",
            keep_debug_files=args.keep_debug_files if hasattr(args, 'keep_debug_files') else False
        )
        # Export final
        cq.exporters.export(mold, args.output)
        print(f"Moule CadQuery généré : {args.output}")
        engraved_shape_keys = [k for k, v in shape_history.items() if isinstance(k, tuple) and v.get('engraved')]
        ignored_shape_keys = [k for k, v in shape_history.items() if isinstance(k, tuple) and not v.get('engraved')]
        # Génération du SVG de résumé avec couleurs distinctes
        all_shape_keys = [k for k in shape_history if isinstance(k, tuple)]
        if all_shape_keys:
            svg_basename = os.path.splitext(os.path.basename(args.svg))[0]
            debug_dir = f"debug_{svg_basename}"
            summary_path = os.path.join(debug_dir, f'summary_{svg_basename}_final.svg')
            generate_summary_svg(args.svg, all_shape_keys, summary_path, shape_history=shape_history)
            print(f"SVG de résumé final généré : {summary_path}")
        else:
            print("Aucun polygone détecté, pas de SVG de visualisation généré.")
    except ValueError as e:
        print(f"Erreur lors de la génération du moule : {e}")

