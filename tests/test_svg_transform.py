import os
import shutil
import unittest
from moule_svg_cadquery import generate_cadquery_mold , generate_summary_svg
from utils import compare_svg_shapes_registration, print_svg_attrs, svg_to_array_and_save
import numpy as np


class TestSVGTransform(unittest.TestCase):
    def setUp(self):
        # Utilise le SVG complexe fourni
        self.svg_file_path = os.path.abspath(os.path.join("svgs", "Anneaux_imbriqués.svg"))
        self.debug_dir = "debug_Anneaux_imbriqués"

    def tearDown(self):
        # Nettoyage du dossier de debug si besoin
        archive_dir = "debug_archive_Anneaux_imbriqués"
        if os.path.exists(archive_dir):
            shutil.rmtree(archive_dir)
        if os.path.exists(self.debug_dir):
            os.rename(self.debug_dir, archive_dir)

    def test_svg_raster_distance(self):
        # Génère le moule et les SVGs de résumé
        mold, engraved_indices, shape_history = generate_cadquery_mold(
            self.svg_file_path,
            max_dim=178,  # Utilise la taille du viewBox du SVG source
            export_steps=True,
            keep_debug_files=True,
        )
        # Génère le SVG de résumé final avec tous les polygones gravés
        all_shape_keys = [k for k, v in shape_history.items() if isinstance(k, tuple) and v.get('engraved')]
        svg_basename = os.path.splitext(os.path.basename(self.svg_file_path))[0]
        summary_svg_final = os.path.join(self.debug_dir, f'summary_{svg_basename}_final.svg')
        generate_summary_svg(self.svg_file_path, all_shape_keys, summary_svg_final, shape_history=shape_history)
        self.assertTrue(os.path.exists(summary_svg_final), "Le SVG de résumé final n'a pas été généré.")

        # Chemin du SVG aplati (après normalisation et flatten)
        svg_normalized = os.path.join(self.debug_dir, f"{svg_basename}_normalized.svg")

        print_svg_attrs(svg_normalized, "SVG aplati")
        print_svg_attrs(summary_svg_final, "SVG résumé")

        src_png_path = os.path.join(self.debug_dir, "raster_normalized.png")
        summary_png_path = os.path.join(self.debug_dir, "raster_summary_final.png")
        src_arr = svg_to_array_and_save(svg_normalized, src_png_path, size=256)
        summary_arr = svg_to_array_and_save(summary_svg_final, summary_png_path, size=256)

        # Calcule la distance (IoU ou somme des différences)
        intersection = np.logical_and(src_arr, summary_arr).sum()
        union = np.logical_or(src_arr, summary_arr).sum()
        iou = intersection / union if union > 0 else 0

        print(f"IoU (Intersection over Union) : {iou:.3f}")

        # --- Comparaison des polygones SVG par registration (Procrustes) ---
        #print("\n[Analyse registration SVG] Polygones SVG aplati vs résumé :")
        #compare_svg_shapes_registration(svg_normalized, summary_svg_final)

        # Vérifie que la similarité est élevée (IoU > 0.95, différence faible)
        self.assertGreater(iou, 0.95, "La similarité IoU entre le SVG source et aplati est trop faible.")

if __name__ == '__main__':
    unittest.main()
