"""Linux launcher service writes: isolated temporary files, no GPU or systemctl."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('zero_launcher', Path(__file__).resolve().parents[1] / 'scripts/zero_launcher.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='inferzero-service-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'XDG_CONFIG_HOME': str(self.root)}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.args = types.SimpleNamespace(service_name='inferzero', replace_service=False, host='127.0.0.1')
        self.unit = self.root / 'systemd/user/inferzero.service'

    def install(self, cmd=None):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = launcher.install_service(cmd or ['/usr/bin/serverz', '--model', '/models/a.gguf'], self.args)
        self.output = out.getvalue()
        return rc

    def test_fresh_unit(self):
        self.assertEqual(self.install(), 0)
        self.assertIn('ExecStart="/usr/bin/serverz"', self.unit.read_text())
        self.assertFalse(list(self.unit.parent.glob('.inferzero-*')))

    def test_existing_unit_unchanged(self):
        self.unit.parent.mkdir(parents=True)
        self.unit.write_bytes(b'private Pi unit\n')
        self.assertEqual(self.install(), 1)
        self.assertEqual(self.unit.read_bytes(), b'private Pi unit\n')
        self.assertIn('--replace-service', self.output)

    def test_explicit_replacement(self):
        self.assertEqual(self.install(), 0)
        self.args.replace_service = True
        self.assertEqual(self.install(['/usr/bin/other']), 0)
        self.assertIn('ExecStart="/usr/bin/other"', self.unit.read_text())

    @unittest.skipIf(os.name == 'nt', 'symlink permissions vary on Windows')
    def test_replacement_does_not_follow_symlink(self):
        target = self.root / 'private.txt'
        target.write_bytes(b'keep me')
        self.unit.parent.mkdir(parents=True)
        self.unit.symlink_to(target)
        self.assertEqual(self.install(), 1)
        self.args.replace_service = True
        self.assertEqual(self.install(), 0)
        self.assertFalse(self.unit.is_symlink())
        self.assertEqual(target.read_bytes(), b'keep me')

    def test_separate_name_and_printed_commands(self):
        self.args.service_name = 'inferzero-package-test.service'
        self.assertEqual(self.install(), 0)
        self.assertFalse(self.unit.exists())
        self.assertTrue(self.unit.with_name('inferzero-package-test.service').exists())
        self.assertIn('restart inferzero-package-test', self.output)

    def test_bad_names_rejected(self):
        for name in ['', '../pi', '/tmp/pi', 'pi\n', 'pi unit', '-pi']:
            self.args.service_name = name
            self.assertEqual(self.install(), 1, name)
        self.assertFalse(self.unit.parent.exists())

    def test_line_breaks_in_command_rejected(self):
        self.assertEqual(self.install(['/bin/server\nExecStartPre=/bin/false']), 1)
        self.assertFalse(self.unit.parent.exists())

    def test_line_breaks_in_cache_environment_rejected(self):
        os.environ['HF_HOME'] = '/cache\nExecStartPre=/bin/false'
        self.assertEqual(self.install(), 1)
        self.assertFalse(self.unit.parent.exists())

    def test_systemd_specifiers_escaped(self):
        self.assertEqual(self.install(['/bin/serverz', 'a$b%"c\\d']), 0)
        self.assertIn('"a$$b%%\\"c\\\\d"', self.unit.read_text())

    def test_racing_writer_preserved(self):
        def raced(src, dst):
            Path(dst).write_bytes(b'other writer')
            raise FileExistsError(dst)
        with patch.object(launcher.os, 'link', side_effect=raced):
            self.assertEqual(self.install(), 1)
        self.assertEqual(self.unit.read_bytes(), b'other writer')
        self.assertFalse(list(self.unit.parent.glob('.inferzero-*')))


if __name__ == '__main__':
    unittest.main()
