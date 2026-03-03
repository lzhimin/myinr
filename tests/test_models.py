import pytest
import torch
from inr.models.siren import SIREN, SineLayer
from inr.models.nerf import NeRF, PositionalEncoding
from inr.models.instant_ngp import InstantNGP, HashEncoder


BATCH = 16


# ── SIREN ─────────────────────────────────────────────────────────────────────

class TestSineLayer:
    def test_output_shape(self):
        layer = SineLayer(3, 64, is_first=True)
        x = torch.rand(BATCH, 3)
        assert layer(x).shape == (BATCH, 64)

    def test_first_layer_init_bounds(self):
        in_features = 3
        layer = SineLayer(in_features, 64, is_first=True)
        bound = 1.0 / in_features
        assert layer.linear.weight.abs().max().item() <= bound + 1e-6

    def test_hidden_layer_init_bounds(self):
        import math
        in_features = 64
        omega_0 = 30.0
        layer = SineLayer(in_features, 64, omega_0=omega_0, is_first=False)
        bound = math.sqrt(6.0 / in_features) / omega_0
        assert layer.linear.weight.abs().max().item() <= bound + 1e-6


class TestSIREN:
    def test_output_shape_scalar(self):
        model = SIREN(in_features=3, out_features=1, hidden_features=32, hidden_layers=2)
        x = torch.rand(BATCH, 3)
        assert model(x).shape == (BATCH, 1)

    def test_output_shape_vector(self):
        model = SIREN(in_features=3, out_features=4, hidden_features=32, hidden_layers=2)
        x = torch.rand(BATCH, 3)
        assert model(x).shape == (BATCH, 4)

    def test_output_is_finite(self):
        model = SIREN(in_features=3, out_features=1, hidden_features=32, hidden_layers=2)
        x = torch.rand(BATCH, 3)
        assert torch.isfinite(model(x)).all()

    def test_gradients_flow(self):
        model = SIREN(in_features=3, out_features=1, hidden_features=32, hidden_layers=2)
        x = torch.rand(BATCH, 3)
        loss = model(x).sum()
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None


# ── NeRF ──────────────────────────────────────────────────────────────────────

class TestPositionalEncoding:
    def test_output_dim(self):
        enc = PositionalEncoding(num_freqs=4)
        x = torch.rand(BATCH, 3)
        out = enc(x)
        assert out.shape == (BATCH, enc.out_dim(3))

    def test_output_dim_no_input(self):
        enc = PositionalEncoding(num_freqs=4, include_input=False)
        x = torch.rand(BATCH, 3)
        out = enc(x)
        assert out.shape == (BATCH, enc.out_dim(3))

    def test_output_is_finite(self):
        enc = PositionalEncoding(num_freqs=4)
        x = torch.rand(BATCH, 3)
        assert torch.isfinite(enc(x)).all()


class TestNeRF:
    def test_output_keys(self):
        model = NeRF(pos_freqs=4, dir_freqs=2, hidden_features=32, hidden_layers=4, skip_layer=2)
        pos = torch.rand(BATCH, 3)
        dirs = torch.rand(BATCH, 3)
        out = model(pos, dirs)
        assert "sigma" in out and "rgb" in out

    def test_sigma_shape(self):
        model = NeRF(pos_freqs=4, dir_freqs=2, hidden_features=32, hidden_layers=4, skip_layer=2)
        out = model(torch.rand(BATCH, 3), torch.rand(BATCH, 3))
        assert out["sigma"].shape == (BATCH, 1)

    def test_rgb_shape(self):
        model = NeRF(pos_freqs=4, dir_freqs=2, hidden_features=32, hidden_layers=4, skip_layer=2)
        out = model(torch.rand(BATCH, 3), torch.rand(BATCH, 3))
        assert out["rgb"].shape == (BATCH, 3)

    def test_sigma_positive(self):
        model = NeRF(pos_freqs=4, dir_freqs=2, hidden_features=32, hidden_layers=4, skip_layer=2)
        out = model(torch.rand(BATCH, 3), torch.rand(BATCH, 3))
        assert (out["sigma"] >= 0).all()

    def test_rgb_range(self):
        model = NeRF(pos_freqs=4, dir_freqs=2, hidden_features=32, hidden_layers=4, skip_layer=2)
        out = model(torch.rand(BATCH, 3), torch.rand(BATCH, 3))
        assert (out["rgb"] >= 0).all() and (out["rgb"] <= 1).all()

    def test_without_viewdirs(self):
        model = NeRF(pos_freqs=4, hidden_features=32, hidden_layers=4, skip_layer=2, use_viewdirs=False)
        out = model(torch.rand(BATCH, 3))
        assert out["sigma"].shape == (BATCH, 1)
        assert out["rgb"].shape == (BATCH, 3)

    def test_gradients_flow(self):
        model = NeRF(pos_freqs=4, dir_freqs=2, hidden_features=32, hidden_layers=4, skip_layer=2)
        out = model(torch.rand(BATCH, 3), torch.rand(BATCH, 3))
        (out["sigma"].sum() + out["rgb"].sum()).backward()
        for param in model.parameters():
            assert param.grad is not None


# ── Instant-NGP ───────────────────────────────────────────────────────────────

class TestHashEncoder:
    def test_output_shape(self):
        enc = HashEncoder(n_levels=4, n_features_per_level=2, log2_hashmap_size=10,
                          base_resolution=4, finest_resolution=32)
        x = torch.rand(BATCH, 3)
        out = enc(x)
        assert out.shape == (BATCH, 4 * 2)

    def test_output_is_finite(self):
        enc = HashEncoder(n_levels=4, n_features_per_level=2, log2_hashmap_size=10,
                          base_resolution=4, finest_resolution=32)
        x = torch.rand(BATCH, 3)
        assert torch.isfinite(enc(x)).all()


class TestInstantNGP:
    def test_output_shape(self):
        model = InstantNGP(n_levels=4, n_features_per_level=2, log2_hashmap_size=10,
                           base_resolution=4, finest_resolution=32,
                           hidden_features=32, hidden_layers=2, out_features=4)
        x = torch.rand(BATCH, 3)
        assert model(x).shape == (BATCH, 4)

    def test_output_is_finite(self):
        model = InstantNGP(n_levels=4, n_features_per_level=2, log2_hashmap_size=10,
                           base_resolution=4, finest_resolution=32,
                           hidden_features=32, hidden_layers=2, out_features=4)
        x = torch.rand(BATCH, 3)
        assert torch.isfinite(model(x)).all()

    def test_scalar_output(self):
        model = InstantNGP(n_levels=4, n_features_per_level=2, log2_hashmap_size=10,
                           base_resolution=4, finest_resolution=32,
                           hidden_features=32, hidden_layers=2, out_features=1)
        x = torch.rand(BATCH, 3)
        assert model(x).shape == (BATCH, 1)

    def test_gradients_flow(self):
        model = InstantNGP(n_levels=4, n_features_per_level=2, log2_hashmap_size=10,
                           base_resolution=4, finest_resolution=32,
                           hidden_features=32, hidden_layers=2, out_features=1)
        x = torch.rand(BATCH, 3)
        model(x).sum().backward()
        for param in model.parameters():
            assert param.grad is not None
