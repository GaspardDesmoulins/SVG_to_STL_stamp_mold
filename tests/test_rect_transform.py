import os
import shutil
import unittest
import numpy as np
from moule_svg_cadquery import generate_cadquery_mold, generate_summary_svg
from xml.etree import ElementTree as ET
from utils import print_svg_attrs, svg_to_array_and_save

class TestSVGRectTransform(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_temp_rect_transform"
        self.svg_dir = os.path.join(self.test_dir, "svgs")
        self.debug_dir = os.path.join(self.test_dir, "debug_svg_rect_transform")
        os.makedirs(self.svg_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

        # SVG avec 2 rectangles :
        # - Un rectangle simple
        # - Un rectangle dans 2 groupes imbriqués, chaque groupe ayant une transformation
        # Matrice de translation : [1 0 6; 0 1 5; 0 0 1]
        # Matrice de scale : [2 0 0; 0 0.5 0; 0 0 1]
        # Matrice composée = scale * translate
        self.svg_content = '''<svg width="20" height="20" xmlns="http://www.w3.org/2000/svg">
  <rect x="2" y="1" width="3" height="2" fill="#00f" stroke="#000"/>
  <g transform="matrix(1,0,0,1,6,5)">
    <g transform="matrix(2,0,0,0.5,0,0)">
      <rect x="0" y="0" width="1" height="4" fill="#f00" stroke="#000"/>
    </g>
  </g>
</svg>'''
        self.svg_file_path = os.path.join(self.svg_dir, "test_rect_transform.svg")
        with open(self.svg_file_path, "w") as f:
            f.write(self.svg_content)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        debug_dirs = [self.debug_dir, "debug_test_rect_transform"]
        for debug_dir in debug_dirs:
            if os.path.exists(debug_dir):
                archive_dir = f"debug_archive_{os.path.basename(debug_dir)}"
                if os.path.exists(archive_dir):
                    shutil.rmtree(archive_dir)
                os.rename(debug_dir, archive_dir)

    def test_rectangles_with_transform(self):
        mold, engraved_indices, shape_history = generate_cadquery_mold(
            self.svg_file_path,
            max_dim=100,
            export_steps=True,
            keep_debug_files=True,
        )
        debug_dir = f"debug_{os.path.splitext(os.path.basename(self.svg_file_path))[0]}"
        self.assertTrue(os.path.exists(debug_dir), f"Le dossier de debug '{debug_dir}' n'a pas été créé.")
        summary_svgs = [f for f in os.listdir(debug_dir) if f.startswith("step_") and f.endswith("_summary.svg")]
        self.assertEqual(len(summary_svgs), 2, "Il devrait y avoir 2 fichiers SVG de résumé, un pour chaque étape.")

        # Vérification des SVG de résumé par étape : 1 path attendu par fichier
        for svg_file in summary_svgs:
            svg_path = os.path.join(debug_dir, svg_file)
            tree = ET.parse(svg_path)
            root = tree.getroot()
            summary_group = root.find(".//*[@id='summary_polygons']")
            if summary_group is None:
                ns = {'svg': 'http://www.w3.org/2000/svg'}
                summary_group = root.find(".//svg:g[@id='summary_polygons']", ns)
            self.assertIsNotNone(summary_group, f"Le groupe 'summary_polygons' est manquant dans {svg_file}")
            paths = summary_group.findall("path")
            if not paths:
                ns = {'svg': 'http://www.w3.org/2000/svg'}
                paths = summary_group.findall("svg:path", ns)
            self.assertEqual(len(paths), 1, f"Le fichier {svg_file} devrait contenir un seul path.")
            fill_rule = paths[0].attrib.get('fill-rule', None)
            self.assertEqual(fill_rule, 'evenodd', f"Le path du résumé doit avoir fill-rule='evenodd'.")
            # Vérifie qu'il n'y a pas de <rect> dans le résumé
            rects = root.findall('.//rect')
            self.assertEqual(len(rects), 0, f"Le fichier {svg_file} ne doit pas contenir de <rect>.")

        # Vérification du SVG de résumé final : 2 paths attendus
        svg_basename = os.path.splitext(os.path.basename(self.svg_file_path))[0]
        summary_svg_final = os.path.join(debug_dir, f'summary_{svg_basename}_final.svg')
        self.assertTrue(os.path.exists(summary_svg_final), "Le SVG de résumé final n'a pas été généré.")
        tree = ET.parse(summary_svg_final)
        root = tree.getroot()
        summary_group = root.find(".//*[@id='summary_polygons']")
        if summary_group is None:
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            summary_group = root.find(".//svg:g[@id='summary_polygons']", ns)
        self.assertIsNotNone(summary_group, f"Le groupe 'summary_polygons' est manquant dans le résumé final")
        paths = summary_group.findall("path")
        if not paths:
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            paths = summary_group.findall("svg:path", ns)
        self.assertEqual(len(paths), 2, f"Le fichier résumé final devrait contenir deux paths.")
        for path in paths:
            fill_rule = path.attrib.get('fill-rule', None)
            self.assertEqual(fill_rule, 'evenodd', f"Chaque path du résumé final doit avoir fill-rule='evenodd'.")
        rects = root.findall('.//rect')
        self.assertEqual(len(rects), 0, f"Le fichier résumé final ne doit pas contenir de <rect>.")

        # Chemin du SVG aplati (après normalisation et flatten)
        svg_normalized = os.path.join(debug_dir, f"{svg_basename}_normalized.svg")

        print_svg_attrs(svg_normalized, "SVG aplati")
        print_svg_attrs(summary_svg_final, "SVG résumé")

        src_png_path = os.path.join(debug_dir, "raster_normalized.png")
        summary_png_path = os.path.join(debug_dir, "raster_summary_final.png")
        src_arr = svg_to_array_and_save(svg_normalized, src_png_path, size=256)
        summary_arr = svg_to_array_and_save(summary_svg_final, summary_png_path, size=256)

        # Calcule la distance (IoU ou somme des différences)
        intersection = np.logical_and(src_arr, summary_arr).sum()
        union = np.logical_or(src_arr, summary_arr).sum()
        iou = intersection / union if union > 0 else 0

        print(f"IoU (Intersection over Union) : {iou:.3f}")

if __name__ == '__main__':
    unittest.main()
