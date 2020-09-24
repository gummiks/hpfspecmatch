from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='hpfspecmatch',
      version='0.1.0',
      description='Matching HPF Spectra',
      long_description=readme(),
      url='https://github.com/gummiks/hpfspecmatch/',
      author='Gudmundur Stefansson',
      author_email='gummiks@gmail.com',
      install_requires=['barycorrpy','emcee','lmfit','hpfspec','crosscorr','pyde','astroquery'],
      packages=['hpfspecmatch'],
      license='GPLv3',
      classifiers=['Topic :: Scientific/Engineering :: Astronomy'],
      keywords='HPF Spectra Astronomy',
      include_package_data=True
      )
