from setuptools import setup

setup(
    name="sam_lstm",
    url="https://github.com/SheikSadi/SAM-LSTM-RESNET.git",
    author="SheikSadi",
    author_email="biis.saadi@gmail.com",
    include_package_data=True,
    packages=["sam"],
    # dependencies
    install_requires=["keras==1.1.0", "Theano", "h5py==2.10.0"],
    version="0.0.0",
    description="Python library to generate saliency maps and cropping",
)
