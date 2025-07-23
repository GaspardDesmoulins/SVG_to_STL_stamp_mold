import os
import shutil
import unittest
import cadquery as cq
from xml.etree import ElementTree as ET

# Assurez-vous que le chemin d'accès au module est correct
from moule_svg_cadquery import generate_cadquery_mold, find_parent

class TestGenerateSummarySvg(unittest.TestCase):

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
        """Nettoie les fichiers de test et archive le dossier de debug."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        # Renommage du dossier de debug pour archivage
        debug_dir = "debug_test_two_shapes"
        archive_dir = "debug_archive_test_two_shapes"
        
        # Supprime l'ancien dossier d'archive s'il existe
        if os.path.exists(archive_dir):
            shutil.rmtree(archive_dir)
            
        # Renomme le dossier de debug actuel
        if os.path.exists(debug_dir):
            os.rename(debug_dir, archive_dir)

    def test_summary_svg_per_step(self):
        """
        Vérifie que generate_summary_svg crée un SVG par étape contenant uniquement les polygones de cette étape.
        """
        # Exécution de la fonction principale pour générer le moule et les fichiers de debug
        # Note: export_steps doit être à True pour générer les SVGs intermédiaires
        mold, engraved_indices, shape_history = generate_cadquery_mold(
            self.svg_file_path,
            max_dim=100,
            export_steps=True,
            keep_debug_files=True,
            # Le dossier de debug sera créé par la fonction, pas besoin de le passer
        )

        # Le nom du dossier de debug est déduit du nom du SVG
        debug_dir = "debug_test_two_shapes"
        self.assertTrue(os.path.exists(debug_dir), f"Le dossier de debug '{debug_dir}' n'a pas été créé.")

        # Il devrait y avoir deux groupes de wires, donc deux étapes de gravure
        # Le nom des groupes peut varier (ex: 0, 1 ou 'rect', 'rect')
        # On liste les fichiers de résumé générés
        summary_svgs = [f for f in os.listdir(debug_dir) if f.startswith("step_") and f.endswith("_summary.svg")]
        
        self.assertEqual(len(summary_svgs), 2, "Il devrait y avoir 2 fichiers SVG de résumé, un pour chaque étape.")

        # Vérification du contenu de chaque SVG de résumé
        for svg_file in summary_svgs:
            svg_path = os.path.join(debug_dir, svg_file)
            print(f"Vérification du fichier : {svg_path}")
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # On cherche le groupe contenant les polygones de résumé
            # Le namespace peut être absent si on l'a enregistré comme namespace par défaut
            summary_group = root.find(".//*[@id='summary_polygons']")
            if summary_group is None:
                # Si non trouvé, on réessaye avec le namespace explicite
                ns = {'svg': 'http://www.w3.org/2000/svg'}
                summary_group = root.find(".//svg:g[@id='summary_polygons']", ns)

            self.assertIsNotNone(summary_group, f"Le groupe 'summary_polygons' est manquant dans {svg_file}")

            # Chaque SVG de résumé ne doit contenir qu'un seul polygone
            polygons = summary_group.findall("polygon")
            if not polygons: # Si vide, on réessaye avec le namespace
                ns = {'svg': 'http://www.w3.org/2000/svg'}
                polygons = summary_group.findall("svg:polygon", ns)

            print(f"  -> Trouvé {len(polygons)} polygone(s).")
            self.assertEqual(len(polygons), 1, f"Le fichier {svg_file} devrait contenir un seul polygone, mais en a {len(polygons)}.")
            
            # Vérification que les rectangles du SVG original ne sont pas présents
            rects = root.findall("rect")
            if not rects: # Si vide, on réessaye avec le namespace
                ns = {'svg': 'http://www.w3.org/2000/svg'}
                rects = root.findall("svg:rect", ns)
            print(f"  -> Trouvé {len(rects)} rectangle(s) du SVG original.")
            self.assertEqual(len(rects), 0, f"Le fichier {svg_file} ne devrait pas contenir les rectangles du SVG original.")

if __name__ == '__main__':
    unittest.main()
