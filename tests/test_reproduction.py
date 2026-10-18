
import unittest
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "reproduce_results.py"


class NumericalReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "The reproduction script failed:\n" + result.stderr
            )
        cls.output = result.stdout

    def test_two_house_values(self):
        self.assertIn("R_0 = 7.000000", self.output)
        self.assertIn("R_0 = 8.869409", self.output)
        self.assertIn("R0(1) = 8.55822297802823", self.output)

    def test_five_house_endpoints(self):
        self.assertIn("R0(0) = 3.7198", self.output)
        self.assertIn("R0(20) = 3.4050", self.output)
        self.assertIn("R_E(0) = 3.0542", self.output)
        self.assertIn("R_E(20) = 2.6160", self.output)

    def test_four_house_turning_region(self):
        self.assertIn("R0(0) = 3.8431", self.output)
        self.assertIn("min = 3.7789 at alpha = 0.85", self.output)
        self.assertIn("R0(20) = 3.7837", self.output)


if __name__ == "__main__":
    unittest.main()
