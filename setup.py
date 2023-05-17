from setuptools import setup, find_packages

setup(
    name='CoderSchoolAI',
    version='0.0.1',
    packages=find_packages(),
    description='A Comprehensive Python Library for Creating and Developing Agent AIs.',
    author='Jonathan Koch, ',
    author_email='johnnykoch002@example.com, ',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='sample setuptools development',
    install_requires=['numpy', 'gym'],
)