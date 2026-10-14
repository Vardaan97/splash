import array
import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('quality', Path(__file__).with_name('pf7-quality.py'))
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)

class QualityTests(unittest.TestCase):
    def test_known_kl(self):
        kld, cosine, agrees = quality.compare([math.log(.75), math.log(.25)], [math.log(.5), math.log(.5)])
        self.assertAlmostEqual(kld, .75*math.log(1.5)+.25*math.log(.5), places=12)
        self.assertTrue(agrees)
        self.assertLess(cosine, 1)

    def test_logsumexp_shift(self):
        self.assertLess(quality.compare([10000.,10001.],[-10000.,-9999.])[0], 1e-10)

    def test_reject_truncated_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'capture'
            path.write_bytes(struct.pack('<I',3)+b'\0'*8)
            with self.assertRaises(ValueError): list(quality.rows(path,3))
            path.write_bytes(struct.pack('<Ifff',3,0,float('nan'),1))
            with self.assertRaises(ValueError): list(quality.rows(path,3))
            path.write_bytes(struct.pack('<Ifff',3,0,1,2))
            self.assertEqual(list(quality.rows(path,3))[0],array.array('f',[0,1,2]))
            with self.assertRaises(ValueError): list(quality.rows(path,4))

    def test_reject_missing_contexts(self):
        with self.assertRaises(ValueError):
            quality.evaluate({'vocabulary':248320,'contexts':[]}, Path('.'))

if __name__ == '__main__': unittest.main()
