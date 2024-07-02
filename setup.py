from setuptools import setup, find_namespace_packages

if __name__ == "__main__":
    setup(
        packages=find_namespace_packages(include=['smarts', 'smarts.*']),
        entry_points={"console_scripts": ["scl=cli.cli:scl"]},
    )
