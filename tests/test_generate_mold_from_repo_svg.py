import os
import unittest

# Attempt to import the pipeline; if CadQuery isn't available, we'll skip the test.
try:
    from moule_svg_cadquery import generate_cadquery_mold
    CADQUERY_AVAILABLE = True
except Exception:
    CADQUERY_AVAILABLE = False


@unittest.skipUnless(CADQUERY_AVAILABLE, "CadQuery or pipeline not available in this environment")
class TestGenerateMoldFromRepoSvg(unittest.TestCase):
    def test_generate_mold_default(self):
        """Generate a mold for svgs/Anneaux_imbriqués_addition.svg using the default (classic) mode."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        svg_path = os.path.join(repo_root, 'svgs', 'Anneaux_imbriqués_addition.svg')
        self.assertTrue(os.path.exists(svg_path), f"SVG introuvable: {svg_path}")

        # Run mold generation with debug retention to inspect outputs if needed
        mold, engraved_indices, shape_history = generate_cadquery_mold(
            svg_path,
            max_dim=100,
            export_steps=True,
            keep_debug_files=True,
        )

        self.assertIsNotNone(mold, "Le moule retourné est None")
        # Basic sanity: some shapes should be engraved
        self.assertGreaterEqual(len(engraved_indices), 1, "Aucun groupe gravé détecté")

        # Check debug directory presence
        debug_dir = 'debug_Anneaux_imbriqués_addition'
        self.assertTrue(os.path.exists(debug_dir), f"Le dossier de debug '{debug_dir}' n'a pas été créé.")

        # Check that a final summary SVG exists
        final_summary = os.path.join(debug_dir, 'summary_Anneaux_imbriqués_addition_final.svg')
        self.assertTrue(os.path.exists(final_summary), "Le résumé SVG final est manquant.")

    # Don't rename in test to avoid interference with other tests; keep files for manual inspection


if __name__ == '__main__':
    unittest.main()
