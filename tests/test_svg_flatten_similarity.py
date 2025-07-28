import os
import unittest
import numpy as np
from moule_svg_cadquery import normalize_svg_fill, flatten_svg_transforms
from utils import svg_to_array_and_save, print_svg_attrs

class TestSVGFlattenSimilarity(unittest.TestCase):
    def setUp(self):
        self.svg_file_path = os.path.abspath(os.path.join("svgs", "Anneaux_imbriqués.svg"))
        self.debug_dir = "debug_test_flatten_similarity"
        os.makedirs(self.debug_dir, exist_ok=True)

    def tearDown(self):
        # Déplacement du dossier de debug vers un dossier d'archive
        archive_dir = "debug_archive_flatten_similarity"
        import shutil
        if os.path.exists(archive_dir):
            shutil.rmtree(archive_dir)
        if os.path.exists(self.debug_dir):
            shutil.move(self.debug_dir, archive_dir)

    def test_flattened_similarity(self):
        # Normalisation et flatten
        normalized_svg = normalize_svg_fill(self.svg_file_path, debug_dir=self.debug_dir)
        flattened_svg = flatten_svg_transforms(normalized_svg, debug_dir=self.debug_dir)

        print_svg_attrs(self.svg_file_path, "SVG source")
        print_svg_attrs(flattened_svg, "SVG aplati")

        src_png_path = os.path.join(self.debug_dir, "raster_source.png")
        flattened_png_path = os.path.join(self.debug_dir, "raster_flattened.png")
        src_arr = svg_to_array_and_save(self.svg_file_path, src_png_path, size=256)
        flattened_arr = svg_to_array_and_save(flattened_svg, flattened_png_path, size=256)

        # Calcule la similarité (IoU et différence de pixels)
        intersection = np.logical_and(src_arr, flattened_arr).sum()
        union = np.logical_or(src_arr, flattened_arr).sum()
        iou = intersection / union if union > 0 else 0
        diff = np.abs(src_arr - flattened_arr).sum()

        print(f"IoU (Intersection over Union) : {iou:.3f}")
        print(f"Différence de pixels : {diff}")

        # Vérifie que la similarité est élevée (IoU > 0.95, différence faible)
        self.assertGreater(iou, 0.95, "La similarité IoU entre le SVG source et aplati est trop faible.")
        self.assertLess(diff, 500, "La différence de pixels entre le SVG source et aplati est trop élevée.")

if __name__ == '__main__':
    unittest.main()
