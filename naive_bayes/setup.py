import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='naive_bayes',
    version='0.0.1',
    author='Nina Dukic',
    author_email='nd663@georgetown.edu',
    description='Multinomial Naive Bayes classifier for attributing '
                'presidential speeches to Kennedy or Johnson',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    extras_require={"dev": ["pytest", "flake8", "autopep8"]},
)