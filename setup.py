from setuptools import setup, find_packages

setup(
    name='CoderSchoolAI',
    version='0.0.5',
    packages=find_packages(),
    include_package_data=True,
    description='A Comprehensive Python Library for Creating and Developing Agent AIs.',
    long_description="""
CoderSchoolAI: A Python Module designed for teaching Modern Day AI to Kids.

This package includes a range of educational tools and templates designed to simplify complex concepts and offer improved learning opportunities. 
It enables the exploration of foundational principles of problem-solving through Artificial Intelligence, providing structured guidance for the exploration and 
implementation of theoretical concepts.

Key Features:
 - Learning Curriculum: Our module makes learning programming engaging and fun, turning complex ideas into digestible chunks.
 - Educational Tools and Templates: We provide tools and templates to simplify complex concepts and enhance learning opportunities.
 - Exploration of Foundational Concepts: Our module enables the exploration of foundational principles of problem-solving through Artificial Intelligence.
 - Structural Guidance: We offer structured guidance for exploring and implementing theoretical concepts in a hands-on way.
 - Simplified Machine Learning Principles: Our module breaks down complex machine learning  principles to facilitate understanding and utilization of these fundamental mechanisms.
 - Neural Networks to Neural Blocks: Our module implements a simpler concept called neural blocks. A Neural Block is an auto-generated neural network Sequential that can be trained to learn from data. 
 - Training Agents/Bots: Users can train agents or bots to play games, leveraging principles of data engineering and neural networks.
 
Algorithms Available:
 - Deep-Q w/ Target Network: Deep-Q is a deep reinforcement learning algorithm that uses a target network
 
The goal of this module is to make Agent AI and machine learning more accessible and enjoyable to learn.

Future Enhancements:
 - Adding {Join/Split/Add} support for Creating Network Chains.
 - Adding Support for Neural Block Graph Architectures.
 - Support for multiple agents/bots in a single game.
 - PPO, DDPG, SAC, TD3, PPO2, DDPG2, SAC2

Known Issues: 

Development Environment: 
 - Python 3.6+

Contribution Guidelines:
 - Pull from GitHub: https://github.com/theCoderSchoolTampa/CoderSchoolAI/CoderSchoolAI/blob/master/
 
 - Make Desired Changes to source code
 
 - Document Changelog: 
        Format [Name] - [Date] - [PR #]:
        - [Change]: 
            [Description]
        - [Change]:
            [Description]
            
 - Edit README.md Documentation if Applicable
 
 - Create a Pull Request: https://github.com/theCoderSchoolTampa/CoderSchoolAI/CoderSchoolAI/pulls/
 
 - Grab a Cup o' Joe, and know that I appreciate you for contributing work to this project :D

""",
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