"""调试场景脚本测试。"""

import matplotlib

matplotlib.use("Agg")

from debug_env_scene import create_environment, show_distribution


def test_creates_a_complete_environment():
    environment = create_environment()

    assert environment.region is not None
    assert environment.masters is not None
    assert environment.members is not None


def test_distribution_includes_ground_users():
    environment = create_environment()

    show_distribution(environment, show=False)

    import matplotlib.pyplot as plt

    labels = [collection.get_label() for collection in plt.gca().collections]
    assert "Ground users" in labels
    plt.close()
