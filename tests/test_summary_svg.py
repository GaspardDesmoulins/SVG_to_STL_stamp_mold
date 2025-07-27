import os
import shutil
import unittest
import cadquery as cq
from xml.etree import ElementTree as ET

# Assurez-vous que le chemin d'accès au module est correct
from moule_svg_cadquery import generate_cadquery_mold, find_parent

class TestGenerateSummarySvg(unittest.TestCase):
    def _run_summary_svg_test(self, svg_content, svg_filename, debug_dir_name, element_tag):
        """
        Méthode utilitaire pour tester la génération de SVG de résumé pour différents types de géométries.
        """
        # Création du SVG de test
        svg_file_path = os.path.join(self.svg_dir, svg_filename)
        with open(svg_file_path, "w") as f:
            f.write(svg_content)

        # Exécution de la fonction principale pour générer le moule et les fichiers de debug
        mold, engraved_indices, shape_history = generate_cadquery_mold(
            svg_file_path,
            max_dim=100,
            export_steps=True,
            keep_debug_files=True,
        )

        debug_dir = debug_dir_name
        self.assertTrue(os.path.exists(debug_dir), f"Le dossier de debug '{debug_dir}' n'a pas été créé.")

        summary_svgs = [f for f in os.listdir(debug_dir) if f.startswith("step_") and f.endswith("_summary.svg")]
        self.assertEqual(len(summary_svgs), 2, "Il devrait y avoir 2 fichiers SVG de résumé, un pour chaque étape.")

        for svg_file in summary_svgs:
            svg_path = os.path.join(debug_dir, svg_file)
            print(f"Vérification du fichier : {svg_path}")
            tree = ET.parse(svg_path)
            root = tree.getroot()

            # Recherche du groupe de polygones de résumé
            summary_group = root.find(".//*[@id='summary_polygons']")
            if summary_group is None:
                ns = {'svg': 'http://www.w3.org/2000/svg'}
                summary_group = root.find(".//svg:g[@id='summary_polygons']", ns)

            self.assertIsNotNone(summary_group, f"Le groupe 'summary_polygons' est manquant dans {svg_file}")

            # On accepte soit un unique <polygon>, soit un unique <path> (pour les anneaux ou formes complexes)
            polygons = summary_group.findall("polygon")
            paths = summary_group.findall("path")
            if not polygons:
                ns = {'svg': 'http://www.w3.org/2000/svg'}
                polygons = summary_group.findall("svg:polygon", ns)
            if not paths:
                ns = {'svg': 'http://www.w3.org/2000/svg'}
                paths = summary_group.findall("svg:path", ns)

            total_shapes = len(polygons) + len(paths)
            print(f"  -> Trouvé {len(polygons)} polygone(s) et {len(paths)} path(s).")
            self.assertEqual(total_shapes, 1, f"Le fichier {svg_file} devrait contenir un seul polygone ou path, mais en a {total_shapes}.")

            # Si c'est un path, on vérifie la présence de l'attribut fill-rule pour la visualisation correcte
            if len(paths) == 1:
                fill_rule = paths[0].attrib.get('fill-rule', None)
                self.assertEqual(fill_rule, 'evenodd', f"Le path du résumé doit avoir fill-rule='evenodd' pour une visualisation correcte.")

            # Vérification que les éléments SVG d'origine ne sont pas présents
            elements = root.findall(element_tag)
            if not elements:
                ns = {'svg': 'http://www.w3.org/2000/svg'}
                elements = root.findall(f"svg:{element_tag}", ns)
            print(f"  -> Trouvé {len(elements)} élément(s) '{element_tag}' du SVG original.")
            self.assertEqual(len(elements), 0, f"Le fichier {svg_file} ne devrait pas contenir les éléments '{element_tag}' du SVG original.")


    def setUp(self):
        """Crée les fichiers et dossiers de test nécessaires."""
        self.test_dir = "test_temp"
        self.svg_dir = os.path.join(self.test_dir, "svgs")
        self.debug_dir_base = os.path.join(self.test_dir, "debug")
        os.makedirs(self.svg_dir, exist_ok=True)
        os.makedirs(self.debug_dir_base, exist_ok=True)

        # Création d'un SVG de test simple avec deux rectangles
        self.svg_content = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
    <rect x="10" y="10" width="30" height="30" style="fill:rgb(0,0,255);stroke-width:3;stroke:rgb(0,0,0)" />
    <rect x="50" y="50" width="40" height="40" style="fill:rgb(255,0,0);stroke-width:3;stroke:rgb(0,0,0)" />
</svg>"""
        self.svg_file_path = os.path.join(self.svg_dir, "test_two_shapes.svg")
        with open(self.svg_file_path, "w") as f:
            f.write(self.svg_content)

    def tearDown(self):
        """Nettoie les fichiers de test et archive le dossier de debug avec un nom unique par test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

        # Détermine le nom du dossier de debug utilisé dans ce test
        test_name = self._testMethodName
        debug_dirs = [
            "debug_test_two_shapes",
            "debug_test_two_ellipses",
            "debug_test_two_paths"
        ]
        for debug_dir in debug_dirs:
            if os.path.exists(debug_dir):
                archive_dir = f"debug_archive_{test_name}"
                # Supprime l'ancien dossier d'archive s'il existe
                if os.path.exists(archive_dir):
                    shutil.rmtree(archive_dir)
                os.rename(debug_dir, archive_dir)

    def test_summary_svg_per_step_rect(self):
        """
        Vérifie que generate_summary_svg crée un SVG par étape contenant uniquement les polygones de cette étape.
        """

        # Test pour deux rectangles
        svg_content = """<svg width='100' height='100' xmlns='http://www.w3.org/2000/svg'>\n    <rect x='10' y='10' width='30' height='30' style='fill:rgb(0,0,255);stroke-width:3;stroke:rgb(0,0,0)' />\n    <rect x='50' y='50' width='40' height='40' style='fill:rgb(255,0,0);stroke-width:3;stroke:rgb(0,0,0)' />\n</svg>"""
        self._run_summary_svg_test(svg_content, "test_two_shapes.svg", "debug_test_two_shapes", "rect")

    def test_summary_svg_per_step_ellipses(self):
        """
        Vérifie la génération de SVG de résumé pour deux ellipses.
        """
        svg_content = """<svg width='100' height='100' xmlns='http://www.w3.org/2000/svg'>\n    <ellipse cx='30' cy='30' rx='20' ry='10' style='fill:rgb(0,255,0);stroke-width:3;stroke:rgb(0,0,0)' />\n    <ellipse cx='70' cy='70' rx='15' ry='25' style='fill:rgb(255,0,255);stroke-width:3;stroke:rgb(0,0,0)' />\n</svg>"""
        self._run_summary_svg_test(svg_content, "test_two_ellipses.svg", "debug_test_two_ellipses", "ellipse")

    def test_summary_svg_per_step_paths(self):
        """
        Vérifie la génération de SVG de résumé pour deux paths.
        """
        svg_content = """<svg width='140' height='100' xmlns='http://www.w3.org/2000/svg'>\n    <path d='M10 10 H 90 V 30 H 10 Z' style='fill:rgb(0,255,255);stroke-width:3;stroke:rgb(0,0,0)' />\n    <path d='M20 60 Q 50 10 80 60 T 140 60' style='fill:rgb(255,128,0);stroke-width:3;stroke:rgb(0,0,0)' />\n</svg>"""
        self._run_summary_svg_test(svg_content, "test_two_paths.svg", "debug_test_two_paths", "path")

if __name__ == '__main__':
    unittest.main()
