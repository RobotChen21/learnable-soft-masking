import unittest

try:
    import torch
    from modules.roberta.model import GATLayer, SoftRGATConv, _to_top1_edge
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, 'PyTorch dependencies are not installed')
class ExpertFusionTest(unittest.TestCase):
    def test_parser_edge_formats_share_the_same_top1_representation(self):
        self.assertEqual(_to_top1_edge((1, 2, 4)), (1, 2, 4))
        self.assertEqual(
            _to_top1_edge((1, 2, [4, 7, 3], [0.6, 0.3, 0.1])),
            (1, 2, 4),
        )

    def test_fixed_prior_is_frozen(self):
        layer = SoftRGATConv(4, 4, 19, learnable_prior=False)
        self.assertFalse(layer.lambda_prior.requires_grad)

    def test_learnable_prior_receives_gradients(self):
        layer = SoftRGATConv(4, 4, 19, learnable_prior=True)
        self.assertTrue(layer.lambda_prior.requires_grad)

    def test_hard_variant_uses_original_rgat(self):
        layer = GATLayer(4, 4, 19, 0.0, expert_fusion='original_hard')
        self.assertFalse(layer.use_soft_masking)
        self.assertFalse(hasattr(layer.conv, 'lambda_prior'))

    def test_all_variants_preserve_node_shape(self):
        x = torch.randn(3, 4)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_type = torch.tensor([0, 1, 2])
        edge_prior = torch.ones(3)
        for mode in ('original_hard', 'learnable_soft'):
            layer = GATLayer(4, 4, 19, 0.0, expert_fusion=mode)
            output, _ = layer(x, edge_index, edge_type, edge_prior)
            self.assertEqual(output.shape, x.shape)

if __name__ == '__main__':
    unittest.main()
