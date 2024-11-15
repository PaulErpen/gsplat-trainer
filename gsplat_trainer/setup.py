from setuptools import setup, find_packages

setup(
    name="gsplat_trainer",
    version="0.1",
    packages=find_packages(include=["gsplat_trainer"]),
    test_suite="test_package",
    install_requires=[
        "torch>=2.0.1",
        "numpy>=1.21.1",
        "pandas>=1.3.3",
        "tqdm==2.2.3",
        "wandb==0.17.0",
        "pytest==8.3.3",
        "plyfile==1.1",
        "pillow==11.0.0",
        "torchvision==0.20.1"
    ],
)