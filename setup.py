from setuptools import setup, find_packages

setup(
    name='alphanet',
    version="0.1.2",
    packages=find_packages(include=['alphanet', 'alphanet.*']),
    install_requires=[
        'pyfiglet',
        'rich',
        'ase',
        'matscipy==1.1.1',
        'rdkit',
        'pydantic',
        'scikit-learn',
        'pydantic_settings'
        
    ],
    entry_points={
        "console_scripts": [
            "alpha-train=alphanet.cli:main",
            "alpha-eval=alphanet.eval:evaluate",
            "alpha-conv=alphanet.state_dict:cli",
            "alpha-freeze=alphanet.freeze:freeze_model"
        ]
    },
    python_requires='>=3.8'
)
