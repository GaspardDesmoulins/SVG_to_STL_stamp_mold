import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import main
from moule_svg_cadquery import MoldGenerationError


class TestMainCli(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.svg_path = os.path.join(self.temp_dir.name, "source.svg")
        with open(self.svg_path, "w", encoding="utf-8") as svg_file:
            svg_file.write('<svg xmlns="http://www.w3.org/2000/svg"/>')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_defaults_to_voxelized_mode(self):
        output_path = os.path.join(self.temp_dir.name, "mold.stl")
        with patch.object(main, "generate_cadquery_mold", return_value=("mold", [], {})) as generate, \
             patch.object(main.cq.exporters, "export") as export, \
             patch.object(sys, "argv", ["main.py", "--svg", self.svg_path, "--output", output_path]):
            main.main()

        self.assertEqual(generate.call_args.kwargs["engraving_mode"], "stepped")
        export.assert_called_once_with("mold", output_path)

    def test_classic_mode_is_explicit(self):
        with patch.object(main, "generate_cadquery_mold", return_value=("mold", [], {})) as generate, \
             patch.object(main.cq.exporters, "export"), \
             patch.object(sys, "argv", ["main.py", "--svg", self.svg_path, "--classic"]):
            main.main()

        self.assertEqual(generate.call_args.kwargs["engraving_mode"], "classic")

    def test_missing_svg_is_a_cli_error(self):
        missing_path = os.path.join(self.temp_dir.name, "missing.svg")
        with patch.object(sys, "argv", ["main.py", "--svg", missing_path]):
            with self.assertRaises(SystemExit) as error:
                main.main()

        self.assertEqual(error.exception.code, 2)

    def test_generation_error_is_reported_without_exporting(self):
        generation_error = MoldGenerationError("gravure voxelisée", "masque vide", 2)
        with patch.object(main, "generate_cadquery_mold", side_effect=generation_error), \
             patch.object(main.cq.exporters, "export") as export, \
             patch.object(sys, "argv", ["main.py", "--svg", self.svg_path]):
            with self.assertRaises(SystemExit) as error:
                main.main()

        self.assertEqual(error.exception.code, 1)
        export.assert_not_called()


if __name__ == "__main__":
    unittest.main()
