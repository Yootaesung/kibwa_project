from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read the long description from README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="kibwa-chatbot",
    version="0.1.0",
    author="Kibwa Team",
    author_email="contact@kibwa.com",
    description="한국어 감정 인식 챗봇",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kibwa-chatbot",
    packages=find_packages(include=['chatbot', 'chatbot.*']),
    package_data={
        'chatbot': [
            'config/*.py',
            'templates/*.html',
            'static/*',
            'data/*',
        ]
    },
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Communications :: Chat",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'kibwa-chatbot=chatbot.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
