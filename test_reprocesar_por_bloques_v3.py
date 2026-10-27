"""Pruebas de logica de checkpoints, reanudacion y borrado restringido.
No requieren GDAL ni datos de la ISS.
"""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name('reprocesar_por_bloques_v3.py')
runner = types.ModuleType('reprocesar_refinamiento_v3')
runner.ALGORITHM = 'iss_viirs_masked_v2'
runner.find_frames = lambda case, lo, hi: []
runner.required_inputs = lambda case, frame: []
runner.source_digest = lambda: {'a': 'b'}
runner.validate_corrected = lambda d, f: (d / (f + '_corrected.points')).is_file()
runner.validate_flow = lambda d, f, margin, feather, mode: (d / (f + '_flow.npy')).is_file()
runner.validate_geotiff = lambda d, f: (d / (f + '_rect.tiff')).is_file()
runner.estimate_bytes = lambda case, frames, root, checks: (0, 0, 0)
runner.sha256_file = lambda p: __import__('hashlib').sha256(p.read_bytes()).hexdigest()
def atomic_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(obj))
    os.replace(temp, p)
runner.atomic_json = atomic_json
runner.source_image = lambda folder, frame: folder / (frame + '.JPG')
sys.modules['reprocesar_refinamiento_v3'] = runner
spec = importlib.util.spec_from_file_location('batch_script_under_test', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class BatchTests(unittest.TestCase):
    def test_batches_do_not_include_completed_ids(self):
        ids = [f'ISS067-E-{i}' for i in (327470, 327471, 327472, 327474, 327475, 327476)]
        blocks = mod.batches_of_pending(ids, {ids[2]}, 2)
        self.assertEqual(blocks, [ids[:2], ids[3:5], ids[5:]])
        for b in blocks:
            self.assertNotIn(ids[2], b)

    def make_frame(self, case, root, frame):
        for folder in ('filtered_points', 'geo', 'viirs_cropped_aligned', 'pics'):
            (case / folder).mkdir(parents=True, exist_ok=True)
        for name, folder in (
            (frame+'.points','filtered_points'),
            (frame+'_rect.tiff','geo'),
            (frame+'_pixel_mapping.csv','geo'),
            (frame+'_viirs.tiff','viirs_cropped_aligned'),
            (frame+'_roi.json','viirs_cropped_aligned'),
            (frame+'.JPG','pics'),
        ):
            (case/folder/name).write_bytes(b'data')
        for folder in ('flow', 'corrected_points', 'geo_corrected', 'footprints'):
            (root/folder).mkdir(parents=True, exist_ok=True)
        (root/'flow'/f'{frame}_flow.npy').write_bytes(b'flow')
        (root/'flow'/f'{frame}_flow_valid.npy').write_bytes(b'valid')
        (root/'flow'/f'{frame}_flow_meta.json').write_text('{}')
        (root/'corrected_points'/f'{frame}_corrected.points').write_text('data')
        (root/'corrected_points'/f'{frame}_correction_stats.json').write_text(json.dumps({
            'n_points': 331, 'applied_count': 205, 'valid_flow_count': 246,
            'correction_mag_percentiles': {'p50': .064, 'max': 28.15}
        }))
        (root/'geo_corrected'/f'{frame}_rect.tiff').write_bytes(b'geotiff')
        (root/'geo_corrected'/f'{frame}_pixel_mapping.csv').write_text('data')
        (root/'footprints'/f'iss_geometric_footprint_{mod.frame_number(frame)}.tif').write_bytes(b'mask')

    def test_checkpoint_and_prune_keeps_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)/'case'; root = case/'paper'/'reprocessing_v3'
            frame = 'ISS067-E-327471'
            self.make_frame(case, root, frame)
            (case/'flow').mkdir()
            (case/'flow'/f'{frame}_flow.npy').write_bytes(b'original')
            mod.mark_completed(case, root, frame, 96, 32, 'masked')
            self.assertTrue(mod.check_completed(case, root, frame))
            self.assertEqual(mod.prune_intermediate(root, frame, {327470}), 9)
            self.assertTrue(mod.check_completed(case, root, frame))
            self.assertEqual((case/'flow'/f'{frame}_flow.npy').read_bytes(), b'original')
            self.assertTrue((root/'geo_corrected'/f'{frame}_rect.tiff').exists())
            summary = mod.rebuild_summary(root, [frame], {frame})
            self.assertEqual(summary['second_tps_files'], 1)
            self.assertEqual(summary['flow_arrays_retained'], 0)
            with (root/'resumen_por_frame.csv').open() as fh:
                self.assertIn('second_tps_ready', fh.read())

    def test_keep_selected_dense_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow = root/'flow'; flow.mkdir()
            frame = 'ISS067-E-327470'
            (flow/f'{frame}_flow.npy').write_bytes(b'abc')
            self.assertEqual(mod.prune_intermediate(root, frame, {327470}), 0)
            self.assertTrue((flow/f'{frame}_flow.npy').exists())

    def test_input_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)/'case'; root = case/'paper'/'reprocessing_v3'
            frame = 'ISS067-E-327471'
            self.make_frame(case, root, frame)
            mod.mark_completed(case, root, frame, 96, 32, 'masked')
            (case/'filtered_points'/f'{frame}.points').write_text('modified')
            with self.assertRaisesRegex(RuntimeError, 'INPUTS'):
                mod.check_completed(case, root, frame)

if __name__ == '__main__':
    unittest.main(verbosity=2)
