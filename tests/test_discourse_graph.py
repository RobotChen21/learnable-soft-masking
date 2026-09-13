import unittest

from modules.discourse_graph import build_discourse_edges


class DiscourseGraphTest(unittest.TestCase):
    def test_main_variants_keep_parser_edges_and_priors(self):
        parsed = [(0, 1, 16), (1, 2, 4)]
        self.assertEqual(build_discourse_edges(3, parsed, 'original_hard'),
                         ([[1, 2]], [4], [1.0]))
        self.assertEqual(build_discourse_edges(3, parsed, 'learnable_soft'),
                         ([[2, 1], [1, 2]], [16, 4], [0.5, 1.0]))

    def test_single_utterance_has_no_discourse_edges(self):
        self.assertEqual(build_discourse_edges(2, [], 'learnable_soft'), ([], [], []))


if __name__ == '__main__':
    unittest.main()
