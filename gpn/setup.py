from setuptools import setup


install_requires = [
    "transformers[torch]==4.45.2",
    "tensorboardX==2.6.2.2",
    "datasets==3.6.0",
    "pyarrow",
    "pandas",
    "numpy",
    "torchinfo",
    "biopython",
    "wandb",
    "einops",
    "pandarallel",
    "bioframe",
    "zstandard",
    "zarr",
    "pyBigWig",
    "joblib",
    "scipy",
    "seaborn",
    "jupyter",
    "scikit-learn"
]


setup(
    name='gpn',
    version='0.1.0',
    description='gpn',
    url='http://github.com/songlab-cal/gpn',
    author='Elix Inc addapting Gonzalo Benegas',
    author_email='',
    license='MIT',
    packages=['gpn', 'gpn.msa'],
    zip_safe=False,
    install_requires=install_requires,
)
