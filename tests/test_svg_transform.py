import os
import shutil
import unittest
from xml.etree import ElementTree as ET
from moule_svg_cadquery import generate_cadquery_mold , generate_summary_svg

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

        # Rasterise les deux SVGs sur une grille de pixels
        import cairosvg
        import numpy as np
        from PIL import Image

        def svg_to_array_and_save(svg_path, out_path, size=256):
            """
            Convertit un SVG en image numpy binaire (noir/blanc) et sauvegarde le PNG avec fond blanc.
            - svg_path : chemin du SVG à rasteriser
            - out_path : chemin du PNG à sauvegarder
            - size : taille de la grille (pixels)
            """
            from io import BytesIO
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

        from io import BytesIO
        src_png_path = os.path.join(self.debug_dir, "raster_source.png")
        summary_png_path = os.path.join(self.debug_dir, "raster_summary_final.png")
        src_arr = svg_to_array_and_save(self.svg_file_path, src_png_path, size=256)
        summary_arr = svg_to_array_and_save(summary_svg_final, summary_png_path, size=256)

        # Calcule la distance (IoU ou somme des différences)
        intersection = np.logical_and(src_arr, summary_arr).sum()
        union = np.logical_or(src_arr, summary_arr).sum()
        iou = intersection / union if union > 0 else 0
        diff = np.abs(src_arr - summary_arr).sum()

        print(f"IoU (Intersection over Union) : {iou:.3f}")
        print(f"Différence de pixels : {diff}")
        # On peut fixer un seuil pour la distance acceptable
        self.assertGreater(iou, 0.95, "Le SVG résumé est trop éloigné du SVG source (IoU < 0.95)")

if __name__ == '__main__':
    unittest.main()
