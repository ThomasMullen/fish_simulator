from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description= f.read()

setup(
    name='zf_animator_tsnm',
    version='0.0.1',    
    package_dir={'':'src'},    
    packages=find_packages(where="src"),
    description='Python toolbox to transform zebrafish tracking data to tail animations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ThomasMullen/zf_animator_tsm',
    author='Thomas Soares Mullen',
    author_email='thomasmullen96@gmail.com',
    license='MIT',
    install_requires=[
			'matplotlib ',
			'numpy',
			'h5py',
			'scipy',                     
			'tqdm',                     
			'wheel',                     
			'Pillow',                     
                      ],
    extras_require={
        "dev":["twine>=4.0.2"],
    },
    python_requires=">=3.9",
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development',
        'Topic :: Utilities',
    ],
)
