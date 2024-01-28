from setuptools import setup, find_packages

setup(
    name='lia',
    version='1.0.0',
    description='Librería de IA para abogados',
    author='',
    author_email='info@lia.com',
    url='https://gitlab.com/legalia/backend.git',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 1 - Development/Stable',
        'Intended Audience :: Developers',
        'License :: TBD',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        'langchain',
        'pinecone-client',
        'typing',
    ],
)
