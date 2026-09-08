"""A new solution must work through existing experiment entry points."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.config import load_config


class SolutionExtensionTests(unittest.TestCase):
    def test_registered_solution_owns_cli_training_and_checkpoint_evaluation(self):
        from methods.solution import Solution
        from experiments.cli import parse_config
        from experiments.runner import evaluate
        from experiments.train import main
        calls = []

        class ExampleSolution(Solution):
            def add_arguments(self, parser, *, training):
                parser.add_argument('--example-width', type=int)

            def resolve_config(self, config, args, *, training, restored):
                calls.append(('resolve', training, restored))
                if args.example_width is not None:
                    config = replace(config, training={**config.training, 'width': args.example_width})
                return config

            def validate_config(self, config, *, training):
                calls.append(('validate', training, config.training['width']))

            def build_method(self, config, ordering, flight, scheduling, *, checkpoint, device):
                from methods.compose import CompositeMethod
                calls.append(('build', checkpoint, device))
                return CompositeMethod(ordering, flight, scheduling)

            def create_trainer(self):
                class ExampleTrainer:
                    def train(self, config, device):
                        calls.append(('train', config.training['width'], str(device)))
                        return 'trained'
                return ExampleTrainer()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method = root/'method.toml'
            method.write_text('[method]\nsolution="example"\n[method.components]\n'
                              'ordering="ers"\nflight="stationary"\nscheduling="local"\n')
            config = root/'config.toml'
            config.write_text('[experiment]\nname="example"\nmethod_config='+json.dumps(str(method))+
                              '\nseed=42\nepisodes=1\nslots=1\nusers=4\ndag_nodes=3\ndevice="cpu"\n'
                              'output_root='+json.dumps(str(root/'results'))+'\n')
            checkpoint = root/'model.bin'
            checkpoint.write_bytes(b'example model')
            with patch.dict('methods.factory.SOLUTIONS', {'example': ExampleSolution}):
                arguments = ['--config', str(config), '--example-width', '7',
                             '--training', 'temperature=0.5']
                _, _, resolved = parse_config(arguments, training=True)
                self.assertEqual(resolved.training, {'width': 7, 'temperature': 0.5})
                self.assertEqual(main(arguments), 'trained')
                main(arguments+['--check'])
                _, _, evaluation = parse_config(arguments+['--checkpoint', str(checkpoint)])
                output = evaluate(evaluation)
            metadata = json.loads((output/'metadata.json').read_text())
            self.assertEqual(metadata['status'], 'completed')
            self.assertEqual(output.parents[1].name, 'example_ers_stationary_local')
            self.assertIn(('build', checkpoint, 'cpu'), calls)
            self.assertIn(('train', 7, 'cpu'), calls)
            self.assertIn(('validate', False, 7), calls)
            self.assertIn(('validate', True, 7), calls)

    def test_nonlearning_solution_rejects_checkpoint_before_creating_run(self):
        from experiments.runner import evaluate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(load_config('config/experiments/random.toml'), output_root=root, checkpoint=root/'model.pt')
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                evaluate(config)
            self.assertEqual(list(root.iterdir()), [])

    def test_framework_does_not_import_specific_solutions(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('cli.py', 'runner.py', 'train.py', 'config.py'):
            source = (root/'experiments'/name).read_text(encoding='utf-8')
            self.assertNotIn('methods.solutions.', source, name)
            self.assertNotIn('member_epochs', source, name)
            self.assertNotIn("== 'ppo", source, name)

    def test_training_overrides_and_solution_help(self):
        import contextlib
        import io
        from experiments.cli import parse_config, training_overrides
        self.assertEqual(training_overrides(['rate=0.01', 'enabled=true', 'layers=[16, 32]']),
                         {'rate': 0.01, 'enabled': True, 'layers': [16, 32]})
        with self.assertRaises(ValueError):
            training_overrides(['invalid.key=3'])
        _, _, config = parse_config(['--config', 'config/experiments/ppo.toml',
                                     '--training', 'member_epochs=2', '--training', 'master_epochs=1'], training=True)
        self.assertEqual(config.episodes, 3)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exit:
            parse_config(['--config', 'config/experiments/ppo.toml', '--help'], training=True)
        self.assertEqual(exit.exception.code, 0)
        self.assertIn('--member-epochs', output.getvalue())

    def test_external_legacy_run_config_resolves_to_current_solution(self):
        from experiments.cli import parse_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config('config/experiments/ppo.toml')
            legacy = replace(config, name='ppo_delay', method={**config.method, 'solution': 'ppo_delay'})
            saved = root/'config.json'
            saved.write_text(json.dumps({'experiment': asdict(legacy)}, default=str), encoding='utf-8')
            original = saved.read_bytes()
            for flag, training in (('--resume', True), ('--checkpoint', False)):
                with self.subTest(flag=flag):
                    checkpoint = root/'checkpoints/latest.pt'
                    _, _, restored = parse_config([flag, str(checkpoint)], training=training)
                    self.assertEqual(restored.method['solution'], 'ppo')
                    self.assertEqual(restored.name, 'ppo')
                    self.assertEqual(restored.episodes, 130 if training else 1)
                    self.assertEqual(saved.read_bytes(), original)
