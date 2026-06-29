from datasets.registry import build_benchmark


def test_rotated_mnist_synthetic_builds(tmp_path):
    cfg = dict(
        name="rotated_mnist",
        source="synthetic",
        seed=0,
        n_tasks=3,
        min_angle=0,
        max_angle=180,
        max_train_per_task=60,
        max_test_per_task=30,
        val_fraction=0.2,
    )
    bench = build_benchmark(cfg, data_root=tmp_path)

    assert bench.name == "rotated_mnist"
    assert len(bench.tasks) == 3
    assert bench.input_shape == (1, 28, 28)
    assert bench.n_classes_per_task == 10
    assert bench.multihead is False
    assert [t.name for t in bench.tasks] == ["rot0_0.0", "rot1_90.0", "rot2_180.0"]
    assert len(bench.tasks[0].train) == 48
    assert len(bench.tasks[0].val) == 12
    assert len(bench.tasks[0].test) == 30
