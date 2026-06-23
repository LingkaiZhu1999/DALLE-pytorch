from setuptools import setup, find_packages
exec(open('dalle_pytorch/version.py').read())

setup(
  name = 'dalle-pytorch',
  packages = find_packages(),
  include_package_data = True,
  version = __version__,
  license='MIT',
  description = 'DALL-E - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/dalle-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'text-to-image'
  ],
  install_requires=[
    'axial_positional_embedding',
    'DALL-E',
    'einops>=0.3.2',
    'ftfy',
    'packaging',
    'pillow',
    'regex',
    'rotary-embedding-torch',
    'tokenizers',
    'torch>=2.0',
    'torchvision',
    'tqdm',
    'WebDataset'
  ],
  extras_require={
    'chinese': [
      'transformers',
      'huggingface-hub>=0.34,<1.0',
    ],
    'vqgan': [
      'taming-transformers-rom1504',
    ],
    'yttm': [
      'youtokentome',
    ],
    'all': [
      'transformers',
      'huggingface-hub>=0.34,<1.0',
      'taming-transformers-rom1504',
      'youtokentome',
    ],
  },
  python_requires='>=3.9',
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
  ],
)
