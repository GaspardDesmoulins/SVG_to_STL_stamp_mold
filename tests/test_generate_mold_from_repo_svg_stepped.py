import os
import unittest

# Attempt to import the pipeline; if CadQuery isn't available, we'll skip the test.
try:
    from moule_svg_cadquery import generate_cadquery_mold
    CADQUERY_AVAILABLE = True
except Exception:
    CADQUERY_AVAILABLE = False


@unittest.skipUnless(CADQUERY_AVAILABLE, "CadQuery or pipeline not available in this environment")
class TestGenerateMoldFromRepoSvgStepped(unittest.TestCase):
    def test_generate_mold_stepped(self):
        """Generate a mold for svgs/Anneaux_imbriqués_addition.svg using the stepped mode."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        svg_path = os.path.join(repo_root, 'svgs', 'Anneaux_imbriqués_addition.svg')
        self.assertTrue(os.path.exists(svg_path), f"SVG introuvable: {svg_path}")

        # Run mold generation with stepped raster mode; use coarser pixels for speed in test
        mold, engraved_indices, shape_history = generate_cadquery_mold(
            svg_path,
            max_dim=50,
            export_steps=True,
            keep_debug_files=True,
            engraving_mode="stepped",
            layer_thickness_mm=0.2,
            pixel_size_mm=0.2,
            growth_per_layer_px=1,
        )

        self.assertIsNotNone(mold, "Le moule retourné est None")
        self.assertGreaterEqual(len(engraved_indices), 1, "Aucun groupe gravé détecté (stepped)")

        debug_dir = 'debug_Anneaux_imbriqués_addition'
        self.assertTrue(os.path.exists(debug_dir), f"Le dossier de debug '{debug_dir}' n'a pas été créé (stepped).")

        final_summary = os.path.join(debug_dir, 'summary_Anneaux_imbriqués_addition_final.svg')
        self.assertTrue(os.path.exists(final_summary), "Le résumé SVG final est manquant (stepped).")

    # Keep debug dir as-is for manual inspection; avoid renaming to prevent test interference


if __name__ == '__main__':
    unittest.main()
