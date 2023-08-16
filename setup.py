from setuptools import setup, find_packages

setup(
    name='CoderSchoolAI',
    version='0.0.5',
    packages=find_packages(),
    include_package_data=True,
    description='A Comprehensive Python Library for Creating and Developing Agent AIs.',
    entry_points={
        'console_scripts': [
            'coderschoolai = CoderSchoolAI.cli.cli:main'
        ]
    },
    author='Jonathan Koch, ',
    author_email='johnnykoch02@gmail.com, ',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='ai agents pygame torch setuptools development',
    install_requires=['numpy', 'gymnasium', 'pygame', 'gym'],
)