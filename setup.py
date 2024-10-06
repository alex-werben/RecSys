from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="recommender_system",
    packages=find_packages(),
    version="0.1.0",
    description="MLOps project for building recommender system.",
    author="Alex Werben",
    entry_points={
        "console_scripts": [
            "recommender_system_train = ml_project.train_pipeline:train_pipeline_command"
        ]
    },
    install_requires=required,
    license="MIT",
)
