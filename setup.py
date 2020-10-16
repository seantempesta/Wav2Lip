from setuptools import Extension, setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Wav2Lip",  # Replace with your own username
    version="0.0.1",
    author="K R Prajwal and Rudrabha Mukhopadhyay and Vinay Namboodiri and C V Jawahar",
    author_email="author@example.com",
    description="A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=find_packages(),
    install_requires=[
        'librosa',
        'numpy',
        'torch',
        'opencv-contrib-python',
        'opencv-python',
        'matplotlib',
        'face_alignment',
        'pydub'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    package_data={'': ['data/mouth_mask.npy']}
)
