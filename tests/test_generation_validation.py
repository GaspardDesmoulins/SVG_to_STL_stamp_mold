import unittest

from moule_svg_cadquery import MoldGenerationError, generate_cadquery_mold


class TestGenerationValidation(unittest.TestCase):
    def test_generation_error_includes_context(self):
        error = MoldGenerationError("gravure voxelisée", "masque vide", 3)

        self.assertEqual(error.stage, "gravure voxelisée")
        self.assertEqual(error.group_idx, 3)
        self.assertIn("groupe 3", str(error))

    def test_invalid_engraving_mode_is_rejected_before_reading_svg(self):
        with self.assertRaisesRegex(ValueError, "engraving_mode"):
            generate_cadquery_mold("does-not-need-to-exist.svg", max_dim=50, engraving_mode="invalid")

    def test_invalid_voxel_parameters_are_rejected_before_reading_svg(self):
        with self.assertRaisesRegex(ValueError, "paramètres de gravure voxelisée"):
            generate_cadquery_mold(
                "does-not-need-to-exist.svg",
                max_dim=50,
                engraving_mode="stepped",
                pixel_size_mm=0,
            )


if __name__ == "__main__":
    unittest.main()
